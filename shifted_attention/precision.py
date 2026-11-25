"""Paired precision acceptance checks on the frozen architecture (Implementation E, item 5)."""
import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from .data import atomic_json
from .model import ShiftedAttentionModel


def scalar_reference(module, z, valid):
    """Independent NumPy float64 implementation on one random, small input."""
    def linear(layer, x):
        return x @ layer.weight.detach().cpu().numpy().astype("f8").T
    b,c,p,s = z.shape
    hs = module.config.heads
    q,k,v = (linear(layer,z).reshape(b,c,p,hs,-1) for layer in
             (module.q_proj,module.k_proj,module.v_proj))
    weights = module.R_nonzero.detach().cpu().numpy().astype("f8")
    weights = np.column_stack((1-weights.sum(-1),weights))
    output = np.zeros_like(q)
    scale = np.sqrt(s//hs)
    for bi in range(b):
        for target,t in zip(*np.nonzero(valid[bi])):
            for h in range(hs):
                keys,values = [],[]
                for j in range(c):
                    if j == target:
                        continue
                    live = [(d,w) for d,w in zip(module.config.shifts,weights[h])
                            if 0 <= t+d < p and valid[bi,j,t+d]]
                    if (module.config.common_support and len(live)!=len(module.config.shifts)) or not any(w!=0 for _,w in live):
                        continue
                    keys.append(sum(w*k[bi,j,t+d,h] for d,w in live))
                    values.append(sum(w*v[bi,j,t+d,h] for d,w in live))
                if keys:
                    a = np.asarray(keys) @ q[bi,target,t,h] / scale
                    a = np.exp(a-a.max())
                    output[bi,target,t,h] = (a/a.sum()) @ np.asarray(values)
    output = output.reshape(b,c,p,s)
    if module.gate_proj is not None:
        gate = linear(module.gate_proj,z)
        output *= 1/(1+np.exp(-gate))
    return linear(module.out_proj,output)


def error_statistics(parts, limit):
    x = np.concatenate(parts)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("expected finite errors on nonempty support")
    q = np.quantile(x,[.5,.95,.99])
    return dict(count=int(x.size),mean=float(x.mean()),rmse=float(np.sqrt(np.square(x).mean())),
                p50=float(q[0]),p95=float(q[1]),p99=float(q[2]),maximum=float(x.max()),
                target=limit,count_above_target=int((x>limit).sum()),maximum_within_target=bool(x.max()<=limit))


@torch.inference_mode()
def check_precision(model, device, *, seed=20261126, draws=8):
    if draws<=0 or model.config.variant=="flat":
        raise ValueError("positive sample count and a shifted channel model are required")
    device = torch.device(device)
    if device.type=="cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("native CUDA bf16 is required")
    fp32 = copy.deepcopy(model).to(device=device,dtype=torch.float32).eval()
    fp64 = copy.deepcopy(fp32).double()
    cfg = fp32.config
    rng = np.random.default_rng(seed)
    errors = {kind:{k:[] for k in ("shifted_output","root_logits","anomaly_logits",
                                     "root_probability","anomaly_probability")}
              for kind in ("fp32_vs_fp64","bf16_vs_fp32")}
    masks = {k:0 for k in errors}
    fixtures = []
    for n in range(draws):
        b,c,p = 2,(1 if n%4==3 else 3+n%3),6+n%3
        t = p*cfg.patch_width-1-n%3
        shape = (b,t,c)
        x = rng.normal(size=shape).astype("f4")
        row = np.ones((b,t),dtype=bool)
        row[1,-(cfg.patch_width+1):] = False
        channel = np.ones((b,c),dtype=bool)
        if c>1:
            channel[1,-1]=False
        observed = (rng.random(shape)>.17) & row[:,:,None] & channel[:,None,:]
        observed[:,cfg.patch_width:2*cfg.patch_width,0] = False
        observed[:,0,:] = False
        x[~observed] = 0
        clock = 1e9 + np.cumsum(rng.uniform(.5,1.5,(b,t)),axis=1)
        z = rng.normal(size=(b,c,p,cfg.stream_width)).astype("f4")
        z -= z.mean(-1,keepdims=True)
        z /= np.sqrt(z.var(-1,keepdims=True)+1e-5)
        active = np.pad(observed,((0,0),(0,p*cfg.patch_width-t),(0,0))).transpose(0,2,1).reshape(b,c,p,-1).any(-1)
        tensor = lambda array: torch.from_numpy(array).to(device)
        obs,clock_t,chan,row_t,token = map(tensor,(observed,clock,channel,row,active))
        block = n%cfg.blocks
        reference = scalar_reference(fp64.blocks[block].channel_attention,z.astype("f8"),active)
        outputs = []
        for net,dt,amp in ((fp64,torch.float64,False),(fp32,torch.float32,False),(fp32,torch.float32,True)):
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
                out = net(tensor(x).to(dt),obs,clock_t,chan,row_t)
                shift = net.blocks[block].channel_attention(tensor(z).to(dt),token)
            item = {k:v.detach().double().cpu().numpy() for k,v in out.items()
                    if k in ("root_logits","anomaly_logits")}
            for task in ("root","anomaly"):
                item[task+"_valid"]=out[task+"_valid"].cpu().numpy()
                # Production scoring explicitly casts BF16 logits to FP32 before sigmoid.
                logits=out[task+"_logits"].to(torch.float64 if dt==torch.float64 else torch.float32)
                item[task+"_probability"]=torch.where(out[task+"_valid"],logits.sigmoid(),0.).double().cpu().numpy()
            item["shifted_output"] = shift.double().cpu().numpy()
            outputs.append(item)
        if not np.allclose(outputs[0]["shifted_output"],reference,atol=1e-10,rtol=1e-10):
            raise AssertionError("FP64 channel operator differs from independent scalar reference")
        outputs[0]["shifted_output"] = reference
        for (kind,pool),left,right in zip(errors.items(),outputs,outputs[1:]):
            for task in ("root","anomaly"):
                masks[kind] += int((left[task+"_valid"]!=right[task+"_valid"]).sum())
            for key in pool:
                mask = active if key=="shifted_output" else left[key.split("_")[0]+"_valid"]
                pool[key].append(np.abs(left[key][mask]-right[key][mask]).ravel())
        fixtures.append(dict(draw=n,shape=[b,t,c],patches=p,operator_layer=block,valid_tokens=int(active.sum()),
                             observed=int(observed.sum()),normalized_input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    stats = {kind:{name:error_statistics(x,1e-5 if kind=="fp32_vs_fp64" else 2e-2)
                   for name,x in group.items()} for kind,group in errors.items()}
    fingerprint=hashlib.sha256()
    for name,value in sorted(fp32.state_dict().items()):
        fingerprint.update(name.encode("utf8")+b"\0"+value.contiguous().cpu().numpy().tobytes())
    return dict(device=str(device),hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
                pytorch=torch.__version__,cuda_runtime=torch.version.cuda,parameter_count=sum(p.numel() for p in fp32.parameters()),
                fp32_state_sha256=fingerprint.hexdigest(),fp32_matmul_precision=torch.get_float32_matmul_precision(),
                seed=seed,draws=fixtures,error_unit="elementwise absolute difference on common valid support",
                sampling="paired NumPy PCG64 streams; standard normal public values, standardized operator features, "
                    "irregular clocks, partial last patch, random missing entries, a missing channel patch, "
                    "row and channel padding, empty point, singleton every fourth draw",
                references=dict(shifted_output="independent NumPy FP64 scalar loop",
                                logits="identical end-to-end production graph and parameters in FP64",
                                probability="sigmoid on valid coordinates after logit cast"),
                mask_mismatches=masks,errors=stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--seed",type=int,default=20261126)
    p.add_argument("--draws",type=int,default=8)
    args=p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.checkpoint:
        from .evaluation.inference import Predictor
        model=Predictor(args.checkpoint,"cpu").model
    else:
        model=ShiftedAttentionModel()
    report=check_precision(model,args.device,seed=args.seed,draws=args.draws)
    if args.checkpoint:
        report["checkpoint_sha256"]=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report["weights"]="checkpoint" if args.checkpoint else "seeded initialization"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,report)
    print(args.output)


if __name__=="__main__":
    main()
