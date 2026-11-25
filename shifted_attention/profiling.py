"""Measure forward/backward by real corpus width; account separately for dense forward MACs."""
import argparse
from dataclasses import asdict
from itertools import cycle, islice
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .config import ModelConfig
from .data import Corpus, atomic_json, build_index, collate
from .model import ShiftedAttentionModel


def forward_flops(cfg, batch, rows, channels):
    """Nominal dense GEMM FLOPs (multiply-add=2), including padded patch positions."""
    b,c,w,s,d=batch,channels,cfg.patch_width,cfg.stream_width,cfg.final_width
    p=(rows+w-1)//w
    elements=b*c*p
    flat=cfg.variant=="flat"
    token_width=d if flat else s
    def corrected(inputs,outputs,linear=False):
        return 2*inputs*outputs + (0 if linear else 2*(inputs*outputs+outputs*outputs))
    def mha(q,k):
        return 4*(q+k)*d*d+4*q*k*d
    tokenizer=elements*corrected(3*w,token_width,cfg.linear_tokenizer)
    if flat:
        stack=b*cfg.blocks*(mha(c*p,c*p)+16*c*p*d*d)
    else:
        projections=(48 if cfg.variant=="no_gate" else 50)*elements*s*s
        attention=4*b*(c*p*p+p*c*c)*s
        stack=cfg.blocks*(projections+attention)
    m=cfg.root_seeds
    root=b*c*(mha(m,p)+16*m*d*d+corrected(m*d,d))+b*(mha(c,c)+2*c*d)
    k=cfg.point_seeds
    recovery=2*elements*d*d*(1 if cfg.repeat_patch_points else w)
    point=b*p*w*(mha(k,c)+16*k*d*d+corrected(k*d,d)+2*d)+recovery
    return dict(tokenizer=tokenizer,encoder=stack,root_readout=root,point_readout=point,
                total=tokenizer+stack+root+point)


def profile(dataset,model,widths,repeats,warmup,device):
    from .train import masks,scaled_loss
    if repeats<=0 or warmup<0:
        raise ValueError("Require positive measured windows and nonnegative warmup")
    rows=[]
    model.train()
    for n in widths:
        indices=np.flatnonzero(dataset.table["n"]==n)
        if not len(indices):
            raise ValueError(f"Corpus has no records with width {n}")
        real_rows=0
        flops=0
        elapsed=0.
        distinct=set()
        for rep,idx in enumerate(islice(cycle(indices),warmup+repeats)):
            batch=collate([dataset[idx]])
            m=masks(batch)
            v=tuple(item.to(device) for item in m)
            batch={k:value.to(device) for k,value in batch.items() if k!="index"}
            model.zero_grad(set_to_none=True)
            if device.type=="cuda":
                torch.cuda.synchronize(device)
                if rep==warmup:
                    torch.cuda.reset_peak_memory_stats(device)
            start=perf_counter()
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                result=model(*(batch[k] for k in ("values","observed","time","channel_valid","row_valid")))
            weight=torch.stack([mask.sum() for mask in v]).float().clamp_min(1).reciprocal()
            loss,_=scaled_loss(result,batch,v,weight)
            loss.backward()
            if device.type=="cuda":
                torch.cuda.synchronize(device)
            if rep>=warmup:
                elapsed+=perf_counter()-start
                t=int(batch["row_valid"].sum())
                real_rows+=t
                distinct.add(int(idx))
                flops+=forward_flops(model.config,1,t,n)["total"]
        rows.append(dict(width=n,measured_windows=repeats,warmup_windows=warmup,distinct_measured_windows=len(distinct),
                         real_rows=real_rows,seconds=elapsed,real_rows_per_second=real_rows/elapsed,
                         supervised_forward_gemm_flops=flops,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",type=Path,required=True,help="Path to a published corpus split directory")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--widths",nargs="+",type=int,default=[1,4,8,16,32,64,128])
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--windows",type=int,default=200)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--variant",default="full")
    args=p.parse_args()
    device=torch.device(args.device)
    cfg=ModelConfig(variant=args.variant)
    saved=None
    if args.checkpoint is not None:
        saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        cfg=ModelConfig(**saved["model_config"])
    torch.manual_seed(17)
    model=ShiftedAttentionModel(cfg).to(device)
    if saved is not None:
        model.load_state_dict(saved["model"])
        del saved
    build_index(args.split)
    ds=Corpus(args.split)
    results=profile(ds,model,args.widths,args.windows,args.warmup,device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,dict(config=asdict(cfg),manifest_sha256=ds.meta["manifest_sha256"],
                timing="one window, model forward + joint masked BCE + backward; I/O and optimizer excluded",
                flops="supervised dense forward GEMMs only; one multiply-add counts two; padding included; nonlinear, recomputation and auxiliary passes excluded",
                torch=str(torch.__version__),device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                widths=results))

if __name__=="__main__":
    main()
