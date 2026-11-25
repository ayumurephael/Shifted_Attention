"""Train the joint detector/localizer with a globally normalized two-task loss."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import Corpus, atomic_json, build_index, collate, consumption_order, validation_halves, MODES
from .model import ShiftedAttentionModel
from .resources import ResourceMeter,parameter_ledger


def masks(batch):
    seen = batch["observed"] & batch["channel_valid"][:,None,:] & batch["row_valid"][:,:,None]
    return (batch["channel_valid"] & batch["R_known"] & batch["E"] & seen.any(1),
            batch["row_valid"] & batch["Y_known"] & seen.any(2))


def scaled_loss(output,batch,valid,weights):
    root = torch.nn.functional.binary_cross_entropy_with_logits(
        output["root_logits"].float(),batch["R"].float(),reduction="none")
    point = torch.nn.functional.binary_cross_entropy_with_logits(
        output["anomaly_logits"].float(),batch["Y"].float(),reduction="none")
    sums = torch.stack(((root*valid[0]).sum(),(point*valid[1]).sum()))
    if "reconstruction_sse" in output:
        sums = torch.cat((sums,output["reconstruction_sse"].float().reshape(1)))
    return (sums*weights).sum(),sums.detach()


def reconstruction_coordinates(batch,w,seed,epoch):
    """Independently select each valid patch; draw coordinates do not depend on rank padding."""
    obs=batch["observed"] & batch["channel_valid"][:,None,:] & batch["row_valid"][:,:,None]
    padded=torch.nn.functional.pad(obs.transpose(1,2),(0,(-obs.shape[1])%w))
    cells=padded.unflatten(-1,(-1,w))
    eligible=cells.any(-1)
    chosen=torch.zeros_like(eligible)
    for b,idx in enumerate(batch["index"].tolist()):
        n=int(batch["channel_valid"][b].sum())
        patches=(int(batch["row_valid"][b].sum())+w-1)//w
        key=hashlib.sha256(f"mask|{seed}|{epoch}|{idx}".encode()).digest()
        gen=torch.Generator().manual_seed(int.from_bytes(key[:8],"little"))
        chosen[b,:n,:patches]=torch.rand((n,patches),generator=gen)<.15
    chosen &= eligible
    return chosen,(cells & chosen.unsqueeze(-1)).sum()


def learning_rate(step,total,warmup,peak=3e-4,end=3e-5):
    if step<=warmup:
        return peak*step/warmup
    return end+(peak-end)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))


def rank_info():
    return (dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def reduce_sum(x):
    if dist.is_initialized():
        dist.all_reduce(x)
    return x


def atomic_save(payload,path):
    temp = path.with_name(path.name + f".{os.getpid()}.part")
    torch.save(payload,temp)
    os.replace(temp,path)


def publish_checkpoint(snapshot, path):
    """Atomically publish an alias on filesystems with or without hard links."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    tmp.unlink(missing_ok=True)
    try:
        try:
            os.link(snapshot, tmp)
        except OSError:
            shutil.copyfile(snapshot, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def versions():
    import platform
    files=sorted(Path(__file__).parent.glob("*.py"))
    digest=hashlib.sha256()
    for f in files:
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    return dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
                cuda=torch.version.cuda,source_sha256=digest.hexdigest())


def save_checkpoint(path,model,opt,state,config,device):
    rank,world=rank_info()
    rng = dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state(device) if device.type=="cuda" else None)
    all_rng=[None]*world
    if world>1:
        dist.all_gather_object(all_rng,rng)
    else:
        all_rng[0]=rng
    if rank==0:
        atomic_save(dict(model_config=asdict(config),model=model.state_dict(),optimizer=opt.state_dict(),
                         rng=all_rng,versions=versions(),**state),path)
    barrier()


class RankIndices:
    def __init__(self, records, seed, budget, cursor, rank, world, effective):
        self.records,self.seed,self.budget = records,seed,budget
        self.cursor,self.rank,self.world,self.effective = cursor,rank,world,effective

    def __iter__(self):
        pos=self.cursor
        size=len(self.records)
        epoch=-1
        while True:
            for offset in range(self.rank*self.effective,(self.rank+1)*self.effective):
                generation,idx=divmod(pos+offset,size)
                if generation!=epoch:
                    order=consumption_order(self.records,self.seed,min(size,self.budget),generation)
                    epoch=generation
                yield int(order[idx])
            pos+=self.world*self.effective


def gather_consumed(local_indices, effective):
    """Restore global sampler order from the coordinates actually read on each rank."""
    rank, world = rank_info()
    local = np.asarray(local_indices,dtype=np.uint32).reshape(-1,effective)
    parts = [local]
    if world > 1:
        parts = [None]*world
        dist.all_gather_object(parts,local)
    if rank == 0:
        # [update,rank,window-within-rank] also covers an all-unknown skipped batch.
        return np.stack(parts,axis=1).reshape(-1)
    return np.empty(0,dtype=np.uint32)


def id_sequence_digest(table, indices):
    digest=hashlib.sha256()
    for start in range(0,len(indices),16384):
        ids=table["id"][indices[start:start+16384]].tolist()
        digest.update(b"\n".join(ids)+b"\n")
    return digest.hexdigest()


def prior_indices(table, seed, budget, cursor):
    """Reconstruct the prefix only for older checkpoints without an observed ledger."""
    result=np.empty(cursor,dtype=np.uint32)
    offset=0
    for epoch,start in enumerate(range(0,cursor,len(table))):
        take=min(len(table),cursor-start)
        result[offset:offset+take]=consumption_order(table,seed,min(budget,len(table)),epoch)[:take]
        offset+=take
    return result


def run(args):
    from .validation import validate

    world=int(os.environ.get("WORLD_SIZE","1"))
    local=int(os.environ.get("LOCAL_RANK","0"))
    device=torch.device(args.device if args.device!="cuda" else f"cuda:{local}")
    if device.type=="cuda":
        torch.cuda.set_device(device)
    if world>1:
        dist.init_process_group("nccl" if device.type=="cuda" else "gloo",timeout=timedelta(hours=18))
    rank,world=rank_info()
    meter=ResourceMeter(device,perf_counter())
    if args.global_batch%world or (args.global_batch//world)%args.microbatch:
        raise ValueError("global batch must be divisible by world size and physical microbatch")
    if args.steps<=0 or args.microbatch<=0 or args.interval<=0 or not 0<=args.warmup<=args.steps:
        raise ValueError("invalid training schedule")
    effective=args.global_batch//world
    args.output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        for split in ("train","validation","test"):
            build_index(args.dataset/split)
    barrier()
    train=Corpus(args.dataset/"train")
    val=Corpus(args.dataset/"validation")
    test=Corpus(args.dataset/"test")
    if not args.fixture and (len(train),len(val),len(test))!=(6720000,336000,336000):
        raise ValueError("unexpected corpus counts for the production profile")
    select,cal=validation_halves(val.table,tiny=args.fixture)
    config=ModelConfig(variant=args.variant,patch_chunk_size=args.patch_chunk,source_chunk_size=args.source_chunk,
                       aux_reconstruction=args.aux_reconstruction,
                       patch_width=args.patch_width,shifts=tuple(lag*args.shift_multiplier for lag in ModelConfig.shifts),
                       common_support=args.common_support,linear_tokenizer=args.linear_tokenizer,
                       repeat_patch_points=args.repeat_patch_points,root_seeds=args.root_seeds,point_seeds=args.point_seeds,
                       **(dict(stream_width=12,blocks=1) if args.fixture else {}))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model=ShiftedAttentionModel(config).to(device)
    opt=torch.optim.AdamW(model.decay_parameter_groups(.05),lr=3e-4,betas=(.9,.95),eps=1e-8)
    net=DistributedDataParallel(model,device_ids=[local] if device.type=="cuda" else None,
                                broadcast_buffers=False) if world>1 else model
    contract=dict(dataset=[d.meta["manifest_sha256"] for d in (train,val,test)],seed=args.seed,
                  steps=args.steps,global_batch=args.global_batch,warmup=args.warmup,interval=args.interval,
                  world=world,model_config=asdict(config))
    latest=args.output/"latest.pt"
    best=args.output/"best.pt"
    cursor=step=empty=0
    best_score=None
    consumed=np.empty(0,dtype=np.uint32)
    reconstructed_prefix=0
    counts=np.zeros((129,7,4,3),dtype=np.int64)
    if latest.exists():
        if not args.resume:
            raise FileExistsError(f"{latest} exists; pass --resume")
        saved=torch.load(latest,map_location="cpu",weights_only=False)
        if saved["contract"]!=contract:
            raise ValueError("resume contract differs from saved data, schedule, world or architecture")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        step,cursor,empty,best_score=(saved[k] for k in ("step","cursor","empty","best_score"))
        counts[:]=saved["exposure"]
        if "resources" in saved:
            meter.state.update(saved["resources"])
        else:
            meter.state["unmeasured_prefix_windows"]=cursor
        if rank==0:
            if "consumed_indices" in saved:
                consumed=saved["consumed_indices"]
                reconstructed_prefix=saved.get("reconstructed_prefix",0)
            else:
                consumed=prior_indices(train.table,args.seed,args.steps*args.global_batch,cursor)
                reconstructed_prefix=cursor
            if len(consumed)!=cursor:
                raise ValueError("saved consumption ledger does not match sampler cursor")
        r=saved["rng"][rank]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        if r["cuda"] is not None:
            torch.cuda.set_rng_state(r["cuda"],device)
        del saved
    if rank==0:
        atomic_json(args.output/"parameter-ledger.json",parameter_ledger(model))
        atomic_json(args.output/"run.json",dict(contract=contract,versions=versions(),model_parameters=sum(
            p.numel() for p in model.parameters()),selection_windows=len(select),calibration_windows=len(cal)))
    sampler=RankIndices(train.table,args.seed,args.steps*args.global_batch,cursor,rank,world,effective)
    loader=DataLoader(train,batch_size=args.microbatch,sampler=sampler,collate_fn=collate,
                      pin_memory=device.type=="cuda",num_workers=args.workers,
                      multiprocessing_context="spawn" if args.workers else None,
                      persistent_workers=args.workers>0,generator=torch.Generator().manual_seed(args.seed+rank))
    iterator=iter(loader)
    local_exposure=np.zeros_like(counts)
    local_consumed=[]
    meter.train()
    while step<args.steps:
        microbatches=[next(iterator) for _ in range(effective//args.microbatch)]
        meter.count(microbatches)
        valid_cpu=[masks(batch) for batch in microbatches]
        numbers=sum(torch.stack([mask.sum() for mask in m]) for m in valid_cpu).to(device)
        aux_masks=[None]*len(microbatches)
        if config.aux_reconstruction:
            chosen=[reconstruction_coordinates(b,config.patch_width,args.seed,cursor//len(train))
                    for b in microbatches]
            aux_masks=[m.to(device,non_blocking=True) for m,_ in chosen]
            numbers=torch.cat((numbers,sum(n for _,n in chosen).reshape(1).to(device)))
        reduce_sum(numbers)
        cursor+=args.global_batch
        for batch in microbatches:
            for i,idx in enumerate(batch["index"].tolist()):
                local_consumed.append(idx)
                row=train.table[idx]
                known=(batch["R"][i]&batch["R_known"][i]&batch["E"][i]&batch["channel_valid"][i]).sum().item()
                local_exposure[int(row["n"]),int(row["k"]),int(row["mode"])]+= (1,int(row["k"]),known)
        if not numbers.any():
            empty+=1
            if cursor>=len(train) and step==0:
                raise ValueError("an entire corpus pass contains no supervised coordinates")
            continue
        weights=(world/numbers.float().clamp_min(1))*(numbers>0)
        meter.count_forwards(microbatches,config)
        rate=learning_rate(step+1,args.steps,args.warmup)
        for group in opt.param_groups:
            group["lr"]=rate
        opt.zero_grad(set_to_none=True)
        logsum=torch.zeros_like(weights)
        aux_seconds=torch.zeros((),device=device) if config.aux_reconstruction else None
        for i,(cpu,m_cpu) in enumerate(zip(microbatches,valid_cpu)):
            batch={k:v.to(device,non_blocking=True) for k,v in cpu.items() if k!="index"}
            valid=tuple(m.to(device,non_blocking=True) for m in m_cpu)
            sync=net.no_sync() if world>1 and i+1<len(microbatches) else nullcontext()
            with sync:
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    result=net(*(batch[k] for k in ("values","observed","time","channel_valid","row_valid")),
                               reconstruction_mask=aux_masks[i])
                loss,subtotal=scaled_loss(result,batch,valid,weights)
                if aux_seconds is not None:
                    aux_seconds+=result["reconstruction_forward_seconds"]
                loss.backward()
            logsum+=subtotal
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        step+=1
        if step==1 or step%50==0:
            loss_logged=reduce_sum(logsum)/numbers.clamp_min(1)
            aux_elapsed=float(reduce_sum(aux_seconds))/world if aux_seconds is not None else None
            if rank==0:
                print(json.dumps(dict(step=step,cursor=cursor,lr=rate,bce=loss_logged[:2].tolist(),
                                      reconstruction_mse=float(loss_logged[2]) if config.aux_reconstruction else None,
                                      reconstruction_forward_seconds_per_rank=aux_elapsed,
                                      counts=numbers.tolist(),grad_norm=float(norm))),flush=True)
        if step%args.interval==0 or step==args.steps:
            meter.pause_train()
            summary=validate(model,val,select,args.output/f"selection-{step:06d}.json",device,
                             args.eval_batch,args.workers,purpose="selection")
            score=summary["selection_score"]
            if score is None and not args.fixture:
                raise ValueError("undefined checkpoint-selection statistic")
            wins=best_score is None or (score is not None and score>best_score)
            if wins:
                best_score=score if score is not None else 0.0
            counts+=reduce_sum(torch.as_tensor(local_exposure,device=device)).cpu().numpy()
            local_exposure.fill(0)
            recent=gather_consumed(local_consumed,effective)
            if rank==0:
                consumed=np.concatenate((consumed,recent))
                if len(consumed)!=cursor:
                    raise ValueError("observed consumption count differs from global sampler cursor")
            local_consumed.clear()
            measured=meter.record_job()
            if rank==0:
                atomic_json(args.output/"performance.json",measured)
            state=dict(step=step,cursor=cursor,empty=empty,best_score=best_score,selection=summary,
                       exposure=counts,contract=contract,consumed_indices=consumed,
                       reconstructed_prefix=reconstructed_prefix,resources=dict(meter.state))
            snapshot=args.output/f"step-{step:06d}.pt"
            save_checkpoint(snapshot,model,opt,state,config,device)
            if rank==0:
                for path in ([best,latest] if wins else [latest]):
                    publish_checkpoint(snapshot,path)
            barrier()
            model.train()
            meter.train()
    selected=torch.load(best,map_location="cpu",weights_only=False)
    model.load_state_dict(selected["model"])
    best_step=selected["step"]
    del selected
    calibration=validate(model,val,cal,args.output/"calibration.json",device,args.eval_batch,args.workers,
                         purpose="calibration")
    result=validate(model,test,np.arange(len(test)),args.output/"synthetic-test.json",device,args.eval_batch,
                    args.workers,purpose="test")
    measured=meter.record_job()
    if rank==0:
        atomic_json(args.output/"performance.json",measured)
        exposure=[dict(N=n,K=k,mode=MODES[m],windows=int(row[0]),direct=int(row[1]),known_positive=int(row[2]))
                  for (n,k,m) in zip(*np.nonzero(counts[:,:,:,0])) for row in [counts[n,k,m]]]
        atomic_json(args.output/"consumption.json",[{k:int(v) if isinstance(v,np.integer) else v
                                                    for k,v in r.items()} for r in exposure])
        atomic_json(args.output/"complete.json",dict(step=step,consumed=cursor,best_step=best_step,
                    consumed_id_sha256=id_sequence_digest(train.table,consumed),
                    consumed_id_encoding="one ASCII window ID plus LF per consumed position",
                    reconstructed_prefix=reconstructed_prefix,
                    threshold=calibration["threshold"],root_map=result["root_map"],point_ap=result["point_ap"]))
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--steps",type=int,default=60000)
    p.add_argument("--warmup",type=int,default=3000)
    p.add_argument("--interval",type=int,default=2000)
    p.add_argument("--global-batch",type=int,default=24)
    p.add_argument("--microbatch",type=int,default=4)
    p.add_argument("--eval-batch",type=int,default=1)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--patch-chunk",type=int,default=8)
    p.add_argument("--source-chunk",type=int,default=128)
    p.add_argument("--device",default="cuda")
    p.add_argument("--variant",choices=["full","synchronous","hard_shifts","no_gate","feedback","flat"],default="full")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--common-support",action="store_true")
    p.add_argument("--linear-tokenizer",action="store_true")
    p.add_argument("--repeat-patch-points",action="store_true")
    p.add_argument("--root-seeds",type=int,default=4)
    p.add_argument("--point-seeds",type=int,default=2)
    p.add_argument("--patch-width",type=int,choices=[4,8,16],default=8)
    p.add_argument("--shift-multiplier",type=int,choices=[1,2],default=1)
    p.add_argument("--fixture",action="store_true",help="Explicit tiny test architecture and corpus")
    p.add_argument("--aux-reconstruction",action="store_true",
                   help="Comparison only: an additional masked forward with global observed-point MSE")
    run(p.parse_args())


if __name__=="__main__":
    main()
