"""Measured run resources; checkpointed counters span process restarts."""
from time import perf_counter

import torch
import torch.distributed as dist
from .profiling import forward_flops


def parameter_ledger(model):
    by_module={n:sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
               for n,m in model.named_modules()}
    return dict(total=sum(by_module.values()),direct_by_module={k:v for k,v in by_module.items() if v},
                recursive_by_component={k:sum(p.numel() for p in m.parameters() if p.requires_grad)
                                        for k,m in model.named_children()})


class ResourceMeter:
    def __init__(self,device,start_time=None):
        self.device=device
        self.state=dict(training_rows=0,training_windows=0,training_seconds=0.,training_gpu_hours=0.,
                        supervised_forward_gemm_flops=0,
                        job_seconds=0.,job_gpu_hours=0.,peak_allocated_bytes=0,peak_reserved_bytes=0,
                        unmeasured_prefix_windows=0)
        self.job_start=start_time if start_time is not None else perf_counter()
        self.train_start=None
        self.rows=0
        self.windows=0
        self.flops=0

    def train(self):
        self.train_start=perf_counter()

    def count(self,microbatches):
        for b in microbatches:
            self.rows+=int(b["row_valid"].sum())
            self.windows+=len(b["row_valid"])

    def _sync(self):
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)

    def count_forwards(self,batches,config):
        for b in batches:
            batch,rows,channels=b["values"].shape
            self.flops+=forward_flops(config,batch,rows,channels)["total"]

    def _reduce(self,items,op):
        v=torch.tensor(items,dtype=torch.float64,device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(v,op=op)
        return v.tolist()

    def pause_train(self):
        self._sync()
        elapsed=perf_counter()-self.train_start
        rows,n,gpu=self._reduce([self.rows,self.windows,elapsed if self.device.type=="cuda" else 0.],
                                dist.ReduceOp.SUM)
        self.state["training_rows"]+=int(rows)
        self.state["training_windows"]+=int(n)
        self.state["training_seconds"]+=self._reduce([elapsed],dist.ReduceOp.MAX)[0]
        self.state["training_gpu_hours"]+=gpu/3600
        flops=torch.tensor(self.flops,dtype=torch.int64,device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flops)
        self.state["supervised_forward_gemm_flops"]+=int(flops)
        self.flops=0
        self.rows=self.windows=0
        self.train_start=None

    def record_job(self):
        self._sync()
        now=perf_counter()
        elapsed=now-self.job_start
        self.job_start=now
        cuda=self.device.type=="cuda"
        wall,alloc,res=self._reduce([elapsed,torch.cuda.max_memory_allocated(self.device) if cuda else 0,
                                    torch.cuda.max_memory_reserved(self.device) if cuda else 0],dist.ReduceOp.MAX)
        self.state["job_seconds"]+=wall
        self.state["job_gpu_hours"]+=self._reduce([elapsed if cuda else 0.],dist.ReduceOp.SUM)[0]/3600
        self.state["peak_allocated_bytes"]=max(int(alloc),self.state["peak_allocated_bytes"])
        self.state["peak_reserved_bytes"]=max(int(res),self.state["peak_reserved_bytes"])
        return self.report()

    def report(self):
        wall=self.state["training_seconds"]
        return dict(self.state,training_real_rows_per_second=self.state["training_rows"]/wall if wall else None,
                    device=str(self.device),gpu=torch.cuda.get_device_name(self.device) if self.device.type=="cuda" else None,
                    timing_scope="training includes I/O, gradient accumulation and updates; job additionally includes validation and test; prefix through committed checkpoints only")
