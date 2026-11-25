"""E.5 paired cluster bootstrap. Run python -m shifted_attention.comparison --help."""
import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .data import FAMILIES, atomic_json
from .point_tallies import TALLY_DTYPE, ancestor_code


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def optional_number(x):
    return float(x) if x is not None and x != "" else np.nan


@dataclass
class Panel:
    names: list
    groups: list
    values: np.ndarray
    truth: str
    point_file: Path | None = None
    expected_counts: dict | None = None
    identities: dict | None = None


def panel(records, provenance="", point_file=None, expected_counts=None, keep_identities=False):
    # Pair identifiers, applicability and ground truth before comparing any scores.
    records.sort(key=lambda r:r[0])
    if (not records and not keep_identities) or any(a[0]==b[0] for a,b in zip(records,records[1:])):
        raise ValueError("Empty panel or duplicate evaluation unit")
    h=hashlib.sha256(provenance.encode("utf8"))
    for key,g,v,metadata in records:
        h.update(json.dumps([key,g,metadata],ensure_ascii=False,separators=(",",":")).encode("utf8")+b"\n")
    return Panel([r[0] for r in records],[r[1] for r in records],
                 np.array([optional_number(r[2]) for r in records]),h.hexdigest(),point_file,expected_counts,
                 {r[0]:(r[1],r[3]) for r in records} if keep_identities else None)


def synthetic(path, spec):
    report=read_json(path)
    if report.get("purpose")!="test":
        raise ValueError("Synthetic comparisons require held-out test output")
    rows=[]
    counts={}
    with gzip.open(path.parent/report["event_records"],"rt",encoding="utf8") as f:
        for r in csv.DictReader(f):
            n=int(r["width"])
            family=FAMILIES.index(r["family"])
            g=ancestor_code(n,family,int(r["ancestor"]))
            known,pos=int(r["known_rows"]),int(r["positive_rows"])
            pair=counts.setdefault(g,[0,0])
            pair[0]+=known
            pair[1]+=pos
            metadata=(n,r["sources"],r["mode"],r["known_roots"],r["positive_roots"],known,pos)
            rows.append((r["window_id"],g,r["root_ap"],metadata))
    pf=path.parent/report["ancestor_points"] if spec["metric"]=="point_ap" else None
    return panel(rows,report.get("manifest_sha256") or "",pf,counts)


def tsb(path,spec,mapping):
    rows=[]
    for file in sorted((path/spec["split"]).glob("*.json")):
        r=read_json(file)
        if spec.get("dataset") and spec["dataset"]!=r["dataset"]:
            continue
        source=r["identity"]
        if source["protocol"]!="tsb" or r["split"]!=spec["split"]:
            raise ValueError("Unexpected protocol or split")
        name=r["file"]
        g=mapping.get((r["split"],name),r.get("physical_record") or name)
        m=(source["metric_commit"],source["source_sha256"],r["dataset"],r["n"],r["positive_rows"],
           r["start_row_retained"],r["normalization"],r["period"])
        if spec["metric"] not in r["metrics"]:
            raise ValueError(f"Metric {spec['metric']} absent in {name}")
        rows.append((name,g,r["metrics"][spec["metric"]],m))
    return panel(rows,keep_identities=True)


def real_rca(path,spec):
    rows=[]
    for r in read_json(path):
        if any(r[k]!=spec[k] for k in ("dataset","truth_unit","mode")):
            continue
        ident=(r["release"],r["recording"],r["event"])
        meta=(r["status"],r["start"],r["stop"],sorted(r["targets"]),sorted(r["candidates"]))
        metrics=r.get("metrics")
        if metrics is not None and spec["metric"] not in metrics:
            raise ValueError(f"Unknown metric {spec['metric']}")
        # Preserve unavailable/degenerate events in coverage, including entire missing groups.
        rows.append((ident,r["group"],metrics[spec["metric"]] if metrics else None,meta))
        if spec["truth_unit"]=="device-owner-proxy" and r["group"]!=r["recording"]:
            raise ValueError("Light-tunnel experiment grouping must match the recording")
    return panel(rows)


class ScalarPair:
    """Fixed within-group sums and counts, recombined using each cluster multiplicity."""
    def __init__(self,a,b,index,n_groups,equal_groups):
        ok=np.isfinite(a.values)&np.isfinite(b.values)
        counts=np.bincount(index[ok],minlength=n_groups)
        self.n=counts
        sums=np.stack([np.bincount(index[ok],weights=p.values[ok],minlength=n_groups) for p in (a,b)])
        if equal_groups:
            sums=sums/np.maximum(counts,1)
            self.n=(counts>0).astype(int)
        self.sums=sums
        self.coverage=dict(units=len(ok),left_finite=int(np.isfinite(a.values).sum()),
                           right_finite=int(np.isfinite(b.values).sum()),paired_finite=int(ok.sum()),
                           groups_with_pairs=int((counts>0).sum()))
        self.batch=128

    def values(self,w):
        with np.errstate(divide="ignore",invalid="ignore"):
            return (w @ self.sums.T) / (w @ self.n)[:,None]


class PointAP:
    """Sparse score-by-ancestor counts. Each replicate re-ranks the pooled tied labels."""
    def __init__(self,p,groups,device):
        data=np.load(p.point_file,mmap_mode="r",allow_pickle=False)
        if data.dtype!=TALLY_DTYPE:
            raise ValueError("Unknown point sufficient-statistics schema")
        s=np.unique(data["probability"])
        group=np.searchsorted(groups,data["ancestor"])
        if (group>=len(groups)).any() or not np.array_equal(groups[group],data["ancestor"]):
            raise ValueError("Point artifact has unknown ancestors")
        if (data["positive"]>data["known"]).any() or not np.isfinite(s).all() or (s<0).any() or (s>1).any():
            raise ValueError("Invalid point tally")
        n=np.bincount(group,weights=data["known"],minlength=len(groups))
        pos=np.bincount(group,weights=data["positive"],minlength=len(groups))
        if any(tuple(p.expected_counts[g])!=(n[i],pos[i]) for i,g in enumerate(groups.tolist())):
            raise ValueError("Point counts disagree with held-out window denominator")
        rank=len(s)-1-np.searchsorted(s,data["probability"])
        self.matrices=[]
        self.device=device
        for key in ("known","positive"):
            m=sparse.csr_matrix((data[key].astype(np.float64),(rank,group)),shape=(len(s),len(groups)))
            m.eliminate_zeros()
            if device!="cpu":
                import torch
                m=torch.sparse_csr_tensor(torch.tensor(m.indptr.astype(np.int64),device=device),
                    torch.tensor(m.indices.astype(np.int64),device=device),torch.tensor(m.data,device=device),
                    size=m.shape,device=device)
            self.matrices.append(m)
        self.count=int(n.sum())
        self.positives=int(pos.sum())
        self.batch=max(1,min(128,1_000_000//max(1,len(s))))

    def values(self,w):
        if self.count==0:
            return np.full(len(w),np.nan)
        if self.device=="cpu":
            n,p=(m @ w.T for m in self.matrices)
            denom=p.sum(0)
            with np.errstate(divide="ignore",invalid="ignore"):
                return (p*np.cumsum(p,axis=0)/np.maximum(1,np.cumsum(n,axis=0))).sum(0)/denom
        import torch
        wt=torch.as_tensor(w.T.copy(),device=self.device,dtype=torch.float64)
        n,p=(torch.sparse.mm(m,wt) for m in self.matrices)
        return ((p*p.cumsum(0)/n.cumsum(0).clamp_min(1)).sum(0)/p.sum(0)).cpu().numpy()


class PointPair:
    def __init__(self,a,b,groups,device):
        self.models=[PointAP(x,groups,device) for x in (a,b)]
        m=self.models[0]
        self.coverage=dict(known_points=m.count,positive_points=m.positives,ancestors=len(groups))
        self.batch=min(x.batch for x in self.models)

    def values(self,w):
        return np.stack([x.values(w) for x in self.models],axis=1)


def multiplicities(rng,draws,g):
    ix=rng.integers(g,size=(draws,g))
    return np.bincount((ix+np.arange(draws)[:,None]*g).reshape(-1),
                       minlength=draws*g).reshape(draws,g).astype(np.float64)


def finite_or_none(x):
    return float(x) if np.isfinite(x) else None


def compare(pairs,seeds,spec,replicates=10000,rng_seed=20261126,device="cpu"):
    if spec["kind"]=="tsb":
        common={}
        for pair in pairs:
            for p in pair:
                for k,metadata in p.identities.items():
                    if common.setdefault(k,metadata)!=metadata:
                        raise ValueError(f"TSB physical group, source truth or metric protocol differs for {k}")
        def aligned(p):
            observed=dict(zip(p.names,p.values))
            return panel([(k,g,observed.get(k,np.nan),metadata) for k,(g,metadata) in common.items()])
        pairs=[[aligned(a),aligned(b)] for a,b in pairs]
    reference=pairs[0][0]
    groups,index=np.unique(reference.groups,return_inverse=True)
    index=index.reshape(-1)
    point=spec["kind"]=="synthetic" and spec["metric"]=="point_ap"
    light=spec.get("truth_unit")=="device-owner-proxy"
    engines=[]
    for a,b in pairs:
        if a.truth!=reference.truth or b.truth!=reference.truth:
            raise ValueError("Paired IDs, reference truths, provenance or grouping differ")
        engine=PointPair(a,b,groups,device) if point else ScalarPair(a,b,index,len(groups),light)
        engines.append(engine)
    ones=np.ones((1,len(groups)))
    observed=np.stack([e.values(ones)[0] for e in engines])
    differences=observed[:,0]-observed[:,1]
    est=differences.mean()
    seed_results={str(s):dict(left=finite_or_none(v[0]),right=finite_or_none(v[1]),
                             difference=finite_or_none(diff),coverage=e.coverage)
                  for s,v,diff,e in zip(seeds,observed,differences,engines)}
    result=dict(metric=spec["metric"],groups=len(groups),seed_results=seed_results,estimate=finite_or_none(est),
                bootstrap_seed=rng_seed,requested=replicates,finite=0,interval=None,p_value=None)
    if light:
        # One descriptor per experiment in the original study (six when all are present).
        result["per_experiment"]={}
        for g_index,g in enumerate(groups):
            w=np.zeros_like(ones)
            w[:,g_index]=1
            result["per_experiment"][str(g)]={str(s):[finite_or_none(v) for v in e.values(w)[0]]
                                               for s,e in zip(seeds,engines)}
    if len(groups)<3 or spec.get("descriptive",False) or not np.isfinite(est):
        result["inference"]="descriptive"
        return result
    batch=max(1,min([max(1,1_000_000//len(groups))]+[e.batch for e in engines]))
    rng=np.random.default_rng(rng_seed)
    boots=[]
    for start in range(0,replicates,batch):
        w=multiplicities(rng,min(batch,replicates-start),len(groups))
        by_seed=[]
        for e in engines:
            arm=e.values(w)
            by_seed.append(arm[:,0]-arm[:,1])
        boots.append(np.mean(by_seed,axis=0))
    draws=np.concatenate(boots)
    finite=draws[np.isfinite(draws)]
    result["finite"]=len(finite)
    if len(finite)/replicates < .99:
        result["inference"]="insufficient_finite_replicates"
        return result
    margin=float(spec.get("margin",0))
    alternative=spec.get("alternative","greater")
    deviations=finite-est
    if alternative=="greater":
        extreme=deviations>=est-margin
    elif alternative=="two-sided":
        extreme=np.abs(deviations)>=abs(est-margin)
    else:
        raise ValueError("alternative must be greater or two-sided")
    interval=np.quantile(finite,[.025,.975]).tolist()
    result.update(inference="paired_cluster_bootstrap",interval=interval,
                  p_value=(1+int(extreme.sum()))/(1+len(finite)),margin=margin,alternative=alternative,
                  lower_exceeds_margin=bool(interval[0]>margin),lower_exceeds_zero=bool(interval[0]>0))
    return result


def holm(values):
    """Monotone step-down adjusted p; unavailable predeclared tests keep their slot."""
    present=[k for k,v in values.items() if v is not None]
    present.sort(key=lambda k:values[k])
    adjusted=dict.fromkeys(values)
    largest=0.
    for i,k in enumerate(present):
        largest=max(largest,min(1.,(len(values)-i)*values[k]))
        adjusted[k]=largest
    return adjusted


def run(spec_path,output,device="cpu"):
    base=spec_path.parent
    spec=read_json(spec_path)
    seeds=spec.get("seeds",[17,29,43])
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Training seeds must be distinct")
    mapping={}
    if spec.get("physical_records"):
        with (base/spec["physical_records"]).open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                key=row["split"],row["file"]
                if key in mapping or not row["physical_record"]:
                    raise ValueError("Invalid physical recording map")
                mapping[key]=row["physical_record"]
    report=dict(seeds=seeds,comparisons={})
    families={}
    for c in spec["contrasts"]:
        for a in c["analyses"]:
            key=c["name"]+"/"+a["name"]
            if key in report["comparisons"]:
                raise ValueError("Duplicate contrast/analysis")
            analyses=[(key,a)]
            if a["kind"]=="tsb" and not a.get("dataset"):
                paths={base/spec["systems"][c[arm]][str(seed)]["tsb"]/a["split"]
                       for seed in seeds for arm in ("left","right")}
                sources=sorted({read_json(f)["dataset"] for p in paths for f in p.glob("*.json")})
                analyses.extend((key+"/source/"+s,{**a,"dataset":s}) for s in sources)
            for name,analysis in analyses:
                pairs=[]
                for seed in seeds:
                    pair=[]
                    for arm in ("left","right"):
                        files=spec["systems"][c[arm]][str(seed)]
                        kind=analysis["kind"]
                        if kind=="synthetic":
                            if analysis["metric"] not in ("root_map","point_ap"):
                                raise ValueError("Unknown synthetic endpoint")
                            v=synthetic(base/files["synthetic"],analysis)
                        elif kind=="tsb":
                            v=tsb(base/files["tsb"],analysis,mapping)
                        elif kind=="rca":
                            v=real_rca(base/files["rca"][analysis["panel"]],analysis)
                        else:
                            raise ValueError("Unknown analysis kind")
                        pair.append(v)
                    pairs.append(pair)
                result=compare(pairs,seeds,analysis,device=device)
                result.update(left=c["left"],right=c["right"],kind=kind)
                if kind=="tsb":
                    expected={"U":350,"M":180}[analysis["split"]]
                    if not analysis.get("dataset"):
                        result["expected_Eva_files"]=expected
                        result["complete_finite_Eva"]=all(
                          r["coverage"]["units"]==expected and r["coverage"]["paired_finite"]==expected
                          for r in result["seed_results"].values())
                report["comparisons"][name]=result
                if name==key and a.get("holm_family"):
                    families.setdefault(a["holm_family"],{})[name]=result["p_value"]
    report["holm_families"]={k:holm(v) for k,v in families.items()}
    report["joint_claims"]={}
    for name,items in spec.get("joint_claims",{}).items():
        values=[]
        for key in items:
            r=report["comparisons"][key]
            valid=r.get("interval") is not None and r.get("complete_finite_Eva",True)
            values.append(r["lower_exceeds_margin"] if valid else None)
        report["joint_claims"][name]=dict(endpoints=items,pass_all=all(values) if None not in values else None)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--device",default="cpu",help="cpu or a CUDA device for exact pooled point AP")
    args=p.parse_args()
    run(args.spec,args.output,args.device)

if __name__=="__main__":
    main()
