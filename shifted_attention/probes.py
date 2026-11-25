"""Offline E.6 diagnostics on held-out data. Run python -m shifted_attention.probes --help."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

import numpy as np
import torch

from . import attention as ops
from .data import INPUTS, build_index, Corpus, atomic_json
from .evaluation.inference import Predictor
from .validation import average_precision


def forward_public(model,sample,device):
    batch = {k:sample[k].to(device).unsqueeze(0) for k in INPUTS}
    return model(**batch)


def support(valid,channel_valid,shifts,coeff,common=False):
    """Counts and reasons per receiver [B,C,P,H]. Self is excluded throughout."""
    b,c,p = valid.shape
    device = valid.device
    center = torch.arange(p,device=device)
    eligible = channel_valid.unsqueeze(-1)
    donors = []
    missing = torch.zeros_like(valid)
    boundary = torch.zeros(p,device=device,dtype=torch.bool)
    for shift in shifts:
        ids = center+shift
        in_range = (ids>=0)&(ids<p)
        here = valid.index_select(2,ids.clamp(0,p-1))
        donors.append(here & in_range)
        missing |= ~here & in_range & eligible
        boundary |= ~in_range
    donor = torch.stack(donors,-1)
    slots = ((donor.unsqueeze(3) & (coeff!=0)[None,None,None]).any(-1))
    if common:
        slots &= donor.all(-1,keepdim=True)
    counts = slots.sum(1,keepdim=True)-slots.long()
    missing_other = (missing.sum(1,keepdim=True)-missing.long())>0
    # 0=complete interior, 1=boundary, 2=in-range donor absent, 3=both.
    reasons = boundary.long()[None,None,:]+2*missing_other.long()
    return counts,reasons.expand(b,c,p)


class SampledSummary:
    """Exact count and mean; uniformly sampled values for bounded-memory quantiles."""
    def __init__(self,capacity):
        self.capacity = capacity
        self.count = 0
        self.total = 0.
        self.stored = np.empty(0)

    def add(self,values,rng):
        x = np.asarray(values,dtype=float).ravel()
        if not x.size:
            return
        total = x.size+self.count
        take = min(total,self.capacity)
        nnew = int(rng.hypergeometric(x.size,self.count,take))
        old = rng.choice(self.stored,take-nnew,replace=False) if take-nnew else self.stored[:0]
        new = rng.choice(x,nnew,replace=False) if nnew else x[:0]
        self.stored = np.r_[old,new]
        self.total += x.sum()
        self.count = total

    def result(self):
        if not self.count:
            return dict(count=0,mean=None,quantiles=None)
        return dict(count=self.count,mean=float(self.total/self.count),
                    quantiles=np.quantile(self.stored,[.05,.25,.5,.75,.95]).tolist(),
                    retained=len(self.stored))


REASONS = ("interior","boundary_only","missing_donor_only","boundary_and_missing_donor")


def readout_hooks(stack,model,aggregates,root_weights):
    """Read public-branch seed summaries, and actual QK attention at the RCA PMA."""
    for name,module in (("root",model.root_pool),("point",model.point_pool)):
        def inspect(mod,args,result,kind=name):
            memory,valid=args
            rows=valid.any(-1)
            if rows.any():
                unit=torch.nn.functional.normalize(result[rows].float(),dim=-1)
                total=(unit @ unit.transpose(-1,-2)).sum(0).cpu().numpy()
                if kind not in aggregates:
                    aggregates[kind]=[np.zeros_like(total,dtype=float),0]
                aggregates[kind][0]+=total
                aggregates[kind][1]+=int(rows.sum())
            if kind=="root":
                seed=mod.seeds[None].expand(len(memory),-1,-1).to(memory.dtype)
                att=mod.attention
                q=att._heads(att.q_proj(mod.q_norm(seed))).float()
                k=att._heads(att.k_proj(mod.kv_norm(memory))).float()
                logits=(q @ k.transpose(-1,-2))*q.shape[-1]**-.5
                logits=logits.masked_fill(~valid[:,None,None,:],-torch.inf)
                weights=torch.softmax(torch.where(rows[:,None,None,None],logits,0.),-1)
                root_weights.append((weights[rows].mean(1).cpu().numpy(),rows.cpu().numpy()))
        handle=module.register_forward_hook(inspect)
        stack.callback(handle.remove)


def temporal_mass(weights,sample,w):
    # Stage assignment takes place after the public-only forward pass.
    mask=(sample["Y"] & sample["Y_known"]).numpy()
    if not mask.any() or not weights.size:
        return None
    patch=np.arange(len(mask))//w
    affected=np.zeros(weights.shape[-1],dtype=bool)
    affected[patch[mask]]=True
    indices=np.flatnonzero(affected)
    first,last=indices[[0,-1]]
    stage=np.where(np.arange(len(affected))<first,0,4)
    stage[first:last+1]=5
    stage[indices]=1+np.minimum(2,(3*(indices-first))//max(1,last-first+1))
    return {name:weights[:,:,stage==j].sum(-1).mean(0).tolist() for j,name in
            enumerate(("before","early_positive","middle_positive","late_positive","after","within_event_gaps"))}


class InspectHeads:
    """Observe the real channel SDPA and projection call. Every hook is removed on exit."""
    def __init__(self,model,channel_valid,storage,capacity,rng):
        self.model = model
        self.channel_valid = channel_valid
        self.storage = storage
        self.cap = capacity
        self.rng = rng
        self.active = None

    def __enter__(self):
        self.stack = ExitStack()
        def hook(mod,fn,pre=False):
            handle = (mod.register_forward_pre_hook if pre else mod.register_forward_hook)(fn)
            self.stack.callback(handle.remove)
        for li,block in enumerate(self.model.blocks):
            if not hasattr(block,"channel_attention"):
                continue
            module = block.channel_attention
            def enter(mod,args,layer=li):
                z,valid = args
                self.active = layer
                self.p = 0
                self.b,self.c = z.shape[:2]
                self.valid = valid
                self.counts,self.reason = support(valid,self.channel_valid,mod.config.shifts,
                                                  mod.coefficients(),mod.config.common_support)
            hook(module,enter,True)
            if module.gate_proj is not None:
                def gate(mod,args,result):
                    self.gate = result.sigmoid()
                hook(module.gate_proj,gate)
            def output_pre(mod,args,layer=li,channel=module):
                post = args[0].unflatten(-1,(channel.config.heads,-1))
                q = post.shape[2]
                pre = self.pre.reshape(self.b,q,channel.config.heads,self.c,-1).permute(0,3,1,2,4)
                gating = self.gate.unflatten(-1,(channel.config.heads,-1)) if channel.gate_proj is not None else None
                sl = slice(self.p,self.p+q)
                reason = self.reason[:,:,sl]
                valid = self.valid[:,:,sl]
                for h in range(channel.config.heads):
                    for r,label in enumerate(REASONS):
                        selected = valid & (reason==r)
                        if not selected.any():
                            continue
                        key = (layer,h,label)
                        entry = self.storage.setdefault(key,dict(support=Counter(),
                            pre=SampledSummary(self.cap),post=SampledSummary(self.cap),gate=SampledSummary(self.cap)))
                        counts = self.counts[:,:,sl,h][selected].cpu().numpy()
                        bins,ns = np.unique(counts,return_counts=True)
                        entry["support"].update({int(k):int(v) for k,v in zip(bins,ns)})
                        for name,x in (("pre",pre),("post",post)):
                            entry[name].add(x[:,:,:,h].float().norm(dim=-1)[selected].cpu().numpy(),self.rng)
                        if gating is not None:
                            entry["gate"].add(gating[:,:,:,h][selected].cpu().numpy(),self.rng)
                self.p += q
            hook(module.out_proj,output_pre,True)
            def leave(mod,args,result):
                self.active = None
            hook(module,leave)
        original = ops._attention
        def record(*args,**kwargs):
            value = original(*args,**kwargs)
            if self.active is not None:
                self.pre = value
            return value
        self.stack.enter_context(patch.object(ops,"_attention",record))
        return self

    def __exit__(self,*exc):
        return self.stack.__exit__(*exc)


def basis_sensitivity(module,device):
    """One other-channel sample available solely at one shifted patch."""
    if len(module.config.shifts)<2:
        return []
    mod = copy.deepcopy(module).to(device).eval()
    cfg = mod.config
    center = max(abs(s) for s in cfg.shifts)+1
    p = 2*center+1
    g = 1
    rng = torch.Generator(device="cpu").manual_seed(20261126)
    z = torch.randn((1,2,p,cfg.stream_width),generator=rng).to(device)
    valid = torch.zeros((1,2,p),dtype=torch.bool,device=device)
    valid[0,0,center] = valid[0,1,center+cfg.shifts[g]] = True
    channel = torch.ones((1,2),dtype=torch.bool,device=device)
    output = []
    for eps in (0.,-1e-6,1e-6):
        mod.R_nonzero[0].zero_()
        mod.R_nonzero[0,g-1] = eps
        count,_ = support(valid,channel,cfg.shifts,mod.coefficients(),cfg.common_support)
        value = mod(z,valid)[0,0,center].float()
        if eps == 0.:
            base = value
        output.append(dict(coefficient=eps,source_count=int(count[0,0,center,0]),
                           output_difference_l2=float((value-base).norm())))
    return output


def classify(delta,direct_roots,adjacency,scale,crop,stop,events):
    """Strict peak comparison on the physical crop, including rows lost in acquisition."""
    n = delta.shape[1]
    roots = set(int(a) for a in direct_roots)
    reached,queue = set(roots),list(roots)
    while queue:
        current = queue.pop()
        for child in adjacency[current]:
            if child not in reached:
                queue.append(child)
                reached.add(child)
    other = sorted(reached-roots)
    peaks = delta[crop:stop].max(axis=0)/scale
    peak_root = float(max(peaks[list(roots)]))
    a = min(e["start"] for e in events)
    b = max(e["stop"] for e in events)
    first = np.flatnonzero((delta[crop:stop,other]/scale>=.5).any(1)) if other else []
    from sster_data.episodes import hysteresis
    h,_ = hysteresis(delta.max(1)/scale)
    reopens = np.flatnonzero(h & ~np.r_[False,h[:-1]])
    other_peak = float(max(peaks[other])) if other else None
    return dict(reachable_nonroots=len(other),source_peak=peak_root,symptom_peak=other_peak,
                peak_gap=None if other_peak is None else other_peak-peak_root,
                symptoms_stronger=other_peak is not None and other_peak>peak_root,
                first_symptom_delay=None if not len(first) else int(crop+first[0]-a),
                propagated_tail_in_crop=bool(h[b:stop].any()),
                late_onset_after_last_event=bool(((reopens>=b)&(reopens<stop)).any()))


def private_case(corpus,row,public,cache,source):
    """Returns only audit labels/statistics. Model forward has a separate four-key signature."""
    from sster_data.scm import world
    from sster_data.planning import Episode
    from sster_data.episodes import realize
    uid = row["id"].decode()
    name = Path(corpus.meta["shards"][int(row["shard"])]).name
    if cache.get("name")!=name:
        if "tar" in cache:
            cache["tar"].close()
        cache.update(name=name,tar=tarfile.open(corpus.split/"private"/name,"r:"))
    archive = cache["tar"]
    m = json.load(archive.extractfile(uid+".replay.json"))
    if "physical_trace" in m:
        with np.load(io.BytesIO(archive.extractfile(m["physical_trace"]).read()),allow_pickle=False) as tr:
            delta = tr["delta"]
    elif source=="stored":
        return None
    else:
        ep = Episode(m["split"],m["N"],m["family"],m["ancestor"],m["episode"],m["K"],
                     m["mode"],m["focal"],m["smoke"])
        inp,tgt,_,tr = realize(ep,m["scale"],m["retry"],audit=True)
        order = np.argsort(m["physical_to_exported"])
        for key,x in {**inp,**tgt}.items():
            regenerated = x if key in ("time","Y","Y_known") else np.take(x,order,axis=-1)
            if not np.array_equal(regenerated,public[key].numpy()):
                raise ValueError(f"private replay differs from published {key} for {uid}; use the generating runtime/backend")
        delta = tr["delta"]
    wld = world(m["split"],m["N"],m["family"],m["ancestor"])
    count,parents,delays = wld.args[:3]
    adj = [set() for _ in range(m["N"])]
    for i in range(m["N"]):
        for k in range(count[i]):
            if delays[i,k]>0 and wld.budget[i,k]>0:
                adj[parents[i,k]].add(i)
    roots = {e["target"] for e in m["events"]}
    if {m["physical_to_exported"][i] for i in roots} != set(np.flatnonzero(public["R"].numpy())):
        raise ValueError("private physical ledger disagrees with exported root labels")
    return classify(delta,roots,adj,m["scale"],m["crop_start"],m["crop_start"]+m["w"],m["events"])


def pulse_locations(w,t):
    if t < 4*w:
        return []
    prefix = (t//w-2)*w
    result = [("singleton",prefix+phase,1,t) for phase in range(w)]
    result += [("short",prefix+j,min(3,t-prefix-j),t) for j in range(w)]
    result += [("straddle",prefix+w-1,2,t)]
    tail = min(t,(t//w-1)*w+max(1,w//2))
    if tail%w:
        result.append(("last_partial",tail-1,1,tail))
    return result


def pulses(model,sample,device,metric,seed):
    w = model.config.patch_width
    t = min(len(sample["time"]),16*w)
    base = {k:(sample[k][:t] if k!="channel_valid" else sample[k]) for k in INPUTS}
    rng = np.random.default_rng(seed)
    cases = []
    for name,start,duration,end in pulse_locations(w,t):
        active = base["observed"][start:start+duration].all(0) & base["channel_valid"]
        choices = np.flatnonzero(active.numpy())
        row = dict(kind=name,phase=start%w,onset=start,length=duration,rows=end)
        if not len(choices):
            cases.append(dict(row,status="no_measured_target"))
            continue
        site = int(rng.choice(choices))
        for sign in (-1,1):
            inp = {k:(v[:end] if k!="channel_valid" else v) for k,v in base.items()}
            inp["values"] = inp["values"].clone()
            old = inp["values"][start:start+duration,site].clone()
            inp["values"][start:start+duration,site] = (old+4*sign).clamp(-30,30)
            if torch.equal(old,inp["values"][start:start+duration,site]):
                cases.append(dict(row,status="clipped_no_change",sign=sign))
                continue
            out = forward_public(model,inp,device)
            s = torch.where(out["anomaly_valid"],out["anomaly_logits"].sigmoid(),0)[0].cpu().numpy()
            y = np.zeros(end,dtype=int)
            y[start:start+duration] = 1
            indices = np.flatnonzero(s==s.max())
            peaks = indices[0]
            clock = inp["time"].numpy()
            error = np.abs(clock[peaks]-clock[start:start+duration]).min()
            vus = None if metric is None else float(metric(y,s,w))
            cases.append(dict(row,status="ok",site=site,sign=sign,peak_row=int(peaks),tied_peaks=len(indices),
                              peak_time_error=float(error),peak_row_error=int(min(abs(peaks-np.arange(start,start+duration)))),
                              vus_pr=vus if vus is not None and np.isfinite(vus) else None))
    return cases


def vus_backend(base):
    from .evaluation.assets import metric_source
    path = metric_source(base,"TSB")
    sys.path.insert(0,str(path.resolve()))
    from TSB_AD.evaluation.basic_metrics import generate_curve
    return lambda y,s,w:generate_curve(y.astype(int),s,w,"opt",250)[-1]


def score(model,s,device):
    output = forward_public(model,s,device)
    return torch.where(output["root_valid"],output["root_logits"].sigmoid(),0)[0].cpu().numpy()


def digest(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def run(args):
    data = args.dataset/args.split
    build_index(data)
    corpus = Corpus(data)
    device = torch.device(args.device)
    models = {"primary":Predictor(args.checkpoint,device).model}
    if args.sync_checkpoint:
        models["synchronous"] = Predictor(args.sync_checkpoint,device).model
        if models["synchronous"].config.variant!="synchronous":
            raise ValueError("paired checkpoint must use variant=synchronous")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(corpus),min(args.limit or len(corpus),len(corpus)),replace=False)
    # Forward hooks are limited using independent IDs; sorting improves tar locality.
    attention_ids = set(map(int,indices[:args.attention_windows]))
    clean_ids = [int(i) for i in indices if corpus.table[i]["k"]==0][:args.pulse_backgrounds]
    rng_stats = np.random.default_rng(args.seed+1)
    counts = Counter()
    storage = {}
    readouts = {}
    seed_time = []
    pulse_rows = []
    metric = vus_backend(args.metrics_base) if args.metrics_base else None
    acc = defaultdict(list)
    cache = {}
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        with (args.output/"windows.jsonl").open("w",encoding="utf-8") as out:
            for ix in sorted(map(int,indices)):
                row = corpus.table[ix]
                relevant = row["n"]>=2 and row["mode"]==1
                counts["sampled"] += 1
                if not relevant and ix not in attention_ids and ix not in clean_ids:
                    continue
                sample = corpus[ix]
                uid = row["id"].decode()
                if ix in clean_ids:
                    for arm,model in models.items():
                        pulse_rows.extend(dict(record=uid,arm=arm,**r) for r in
                                          pulses(model,sample,device,metric,args.seed+ix))
                if ix in attention_ids:
                    counts["attention_windows"] += 1
                    root_weights=[]
                    with ExitStack() as hook_scope:
                        readout_hooks(hook_scope,models["primary"],readouts,root_weights)
                        hook_scope.enter_context(InspectHeads(models["primary"],sample["channel_valid"].to(device)[None],
                                      storage,args.reservoir,rng_stats))
                        predicted = score(models["primary"],sample,device)
                    mass=temporal_mass(root_weights[0][0],sample,models["primary"].config.patch_width)
                    if mass is not None:
                        seed_time.append(dict(id=uid,relative_stage_mass=mass))
                else:
                    predicted = None
                if not relevant:
                    continue
                counts["multivariate_process"] += 1
                known = (sample["R_known"] & sample["E"]).numpy()
                root = sample["R"].numpy()
                if not (root[known].any() and (~root[known]).any()):
                    counts["no_evaluable_positive_negative_pair"] += 1
                    continue
                label = private_case(corpus,row,sample,cache,args.private_source)
                if label is None:
                    counts["trace_not_stored"] += 1
                    continue
                counts["classified"] += 1
                if not label["reachable_nonroots"]:
                    counts["no_reachable_nonroot"] += 1
                stat = np.where(sample["observed"].numpy(),abs(sample["values"].numpy()),0).max(0)
                aps = {"max_abs_normalized":average_precision(stat[known],root[known])}
                for arm,model in models.items():
                    scores = predicted if arm=="primary" and predicted is not None else score(model,sample,device)
                    aps[arm] = average_precision(scores[known],root[known])
                for name,selected in (("eligible",True),("symptoms_stronger",label["symptoms_stronger"]),
                                      ("late_onset",label["late_onset_after_last_event"]),
                                      ("propagated_tail",label["propagated_tail_in_crop"])):
                    if selected:
                        counts[name] += 1
                        group=(int(row["n"]),int(row["family"]),int(row["ancestor"]))
                        acc[name].append((uid,group,aps))
                item = dict(id=uid,n=int(row["n"]),family=int(row["family"]),ancestor=int(row["ancestor"]),
                            known_channels=int(known.sum()),known_roots=int(root[known].sum()),
                            observed_fraction=float(sample["observed"].float().mean()),ap=aps,**label)
                out.write(json.dumps(item,allow_nan=False)+"\n")
    finally:
        if "tar" in cache:
            cache["tar"].close()
    heads = []
    for li,block in enumerate(models["primary"].blocks):
        if not hasattr(block,"channel_attention"):
            continue
        mod = block.channel_attention
        c = mod.coefficients().cpu().numpy()
        for h,values in enumerate(c):
            by_reason = {}
            for reason in REASONS:
                ent = storage.get((li,h,reason))
                by_reason[reason] = (None if ent is None else dict(query_count=sum(ent["support"].values()),
                    support_histogram=dict(ent["support"]),**{k:ent[k].result() for k in ("pre","post","gate")}))
            heads.append(dict(layer=li,head=h,shifts=list(mod.config.shifts),lambda_values=values.tolist(),
                              zero=int((values==0).sum()),negative=int((values<0).sum()),
                              tiny_nonzero=int(((np.abs(values)<=1e-6)&(values!=0)).sum()),by_reason=by_reason,
                              zero_epsilon_check=basis_sensitivity(mod,device) if h==0 else None))
    (args.output/"pulses.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in pulse_rows),encoding="utf-8")
    phase_summary = []
    buckets = defaultdict(list)
    for r in pulse_rows:
        buckets[r["arm"],r["kind"],r["phase"]].append(r)
    for (arm,kind,phase),rows in sorted(buckets.items()):
        good = [r for r in rows if r["status"]=="ok"]
        vus = [r["vus_pr"] for r in good if r["vus_pr"] is not None]
        phase_summary.append(dict(arm=arm,kind=kind,phase=phase,cases=len(rows),evaluable=len(good),
            mean_peak_time_error=float(np.mean([r["peak_time_error"] for r in good])) if good else None,
            finite_vus=len(vus),mean_vus_pr=float(np.mean(vus)) if vus else None))
    from .comparison import panel, compare
    summary,paired={},{}
    for name in ("eligible","symptoms_stronger","late_onset","propagated_tail"):
        rows=acc[name]
        counts.setdefault(name,0)
        summary[name]=None
        paired[name]=None
        if not rows:
            continue
        summary[name]={arm:float(np.mean([r[2][arm] for r in rows])) for arm in rows[0][2]}
        def to_panel(arm):
            return panel([(uid,"/".join(map(str,g)),v[arm],None) for uid,g,v in rows],corpus.meta["manifest_sha256"])
        primary=to_panel("primary")
        paired[name]={arm:compare([(primary,to_panel(arm))],["checkpoint_pair"],
                                {"kind":"synthetic","metric":"root_ap"},rng_seed=args.seed)
                      for arm in summary[name] if arm!="primary"}
    report=dict(release=corpus.meta["manifest_sha256"],checkpoint=digest(args.checkpoint),
        synchronous_checkpoint=digest(args.sync_checkpoint) if args.sync_checkpoint else None,
        split=args.split,seed=args.seed,private_source=args.private_source,finite_sample=(len(indices)<len(corpus)),
        counts=counts,root_event_macro_ap=summary,paired_ancestor_bootstrap=paired,heads=heads,pulses=phase_summary,
        pma_cosine={k:{"sets":count,"mean":(total/count).tolist()} for k,(total,count) in readouts.items()},
        root_seed_temporal_mass=seed_time,
        symptom_stratum_contract=("In the frozen direct-state generator every nonroot has zero direct perturbation "
            "and historical Lipschitz row sum below one. With all injections inside the crop, the strict "
            "physical symptom peak cannot exceed the maximum peak at direct positions in exact arithmetic. "
            "An empty stratum has no AP or inferential interval; any floating-point exception needs inspection."),
        gate_quantiles=[.05,.25,.5,.75,.95],gate_quantile_sample_cap=args.reservoir,
        pulse_vus_protocol="TSB generate_curve, version=opt, thre=250, tolerance=patch width"
            if metric is not None else "disabled: supply --metrics-base with the pinned official evaluator",
        peak_tie_rule="first maximum; per-case tie count recorded")
    atomic_json(args.output/"report.json",report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--split",choices=("test","validation"),default="test")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--sync-checkpoint",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--private-source",choices=("replay","stored"),default="replay",
                   help="Reproduce missing physical traces with Data-Generator, or count them as unavailable.")
    p.add_argument("--attention-windows",type=int,default=32)
    p.add_argument("--pulse-backgrounds",type=int,default=8)
    p.add_argument("--reservoir",type=int,default=1024)
    p.add_argument("--limit",type=int,default=0,help="Uniform sample of windows; zero examines the entire split.")
    p.add_argument("--seed",type=int,default=20261126)
    p.add_argument("--metrics-base",type=Path,help="Benchmark assets directory containing pinned TSB sources for pulse VUS-PR.")
    args = p.parse_args()
    if args.limit<0 or args.reservoir<1 or min(args.attention_windows,args.pulse_backgrounds)<0:
        p.error("invalid sampling limits")
    print(json.dumps(run(args)["counts"]))


if __name__=="__main__":
    main()
