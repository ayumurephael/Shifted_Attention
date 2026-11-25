"""Separate event-given rankings, alarm-conditioned localization and outside-panel assertions."""
import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import ndcg_score

from .adapters import segments
from .assets import digest, save_json
from .inference import Predictor
from .baselines import Severity


def ranking(ids, scores, targets, release, run):
    y = np.isin(ids, targets)
    if len(ids)!=len(set(ids)) or not set(targets).issubset(ids) or not y.any() or y.all():
        raise ValueError("Ranking needs exhaustive positive and negative eligible candidates")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid rank scores")
    ties = [hashlib.sha256(f"{release}|{run}|{c}".encode()).hexdigest() for c in ids]
    order = np.lexsort((ties,-scores))
    truth, ranked = y[order], scores[order]
    hits = np.cumsum(truth)
    ends = np.flatnonzero(np.r_[ranked[1:]!=ranked[:-1], True])
    precision, recall = hits[ends]/(ends+1), hits[ends]/sum(y)
    ap = np.dot(np.diff(np.r_[0.,recall]), precision)
    first = np.flatnonzero(truth)[0]+1
    out = {"Hit@1":float(first==1),"MRR":float(1/first),"AP":float(ap)}
    for k in [1,3,5,10]:
        cut = min(k,len(y))
        out[f"Recall@{k}"] = float(hits[cut-1]/sum(y))
        out[f"AC@{k}"] = float(hits[cut-1]/min(k,sum(y)))
        out[f"Avg@{k}"] = float(np.mean([hits[min(j,len(y))-1]/min(j,sum(y)) for j in range(1,k+1)]))
    for percent in (100,150):
        cut = min(len(y),round(percent*int(y.sum())/100))
        out[f"event-HR@{percent}%"] = float(hits[cut-1]/sum(y))
        out[f"event-NDCG@{percent}%"] = float(ndcg_score(y[None,:],scores[None,:],k=cut))
    return out


def overlap(a,b):
    inter = max(0,min(a[1],b[1])-max(a[0],b[0]))
    return inter/(a[1]-a[0]+b[1]-b[0]-inter)


def match_events(events,preds):
    """Return original true-event index -> prediction index. Max cardinality, summed IoU, then chronological ties."""
    records = sorted(enumerate(events),key=lambda x:(x[1]["start"],x[1]["id"]))
    if not events or not preds:
        return {}
    pred_ids = sorted(range(len(preds)),key=lambda j:(preds[j][0],preds[j][1]))
    pairs = np.array([[overlap((e["start"],e["stop"]),preds[j]) for j in pred_ids] for _,e in records])
    pairs = np.where(pairs>=0.1, len(events)+1+pairs, 0.)
    # Drop predictions with no legal edges before repeated tie resolution.
    active = np.flatnonzero(pairs.any(axis=0)).tolist()
    def optimum(i, available):
        a = pairs[i:,available]
        if a.size==0:
            return 0.
        r,c = linear_sum_assignment(a,maximize=True)
        return float(a[r,c].sum())
    matched = {}
    for i,(idx,_) in enumerate(records):
        optimum_now = optimum(i,active)
        for j in active:
            if pairs[i,j]==0:
                continue
            rest = [v for v in active if v!=j]
            if abs(pairs[i,j]+optimum(i+1,rest)-optimum_now) <= 1e-10:
                matched[idx] = pred_ids[j]
                active = rest
                break
    return matched


def aggregate(rows):
    """Equal event weight; for chamber light equal experiments after averaging their windows."""
    groups = defaultdict(list)
    outside = defaultdict(list)
    counts = defaultdict(int)
    for row in rows:
        unit = "|".join(row[k] for k in ("dataset","truth_unit","mode"))
        counts[unit+"|"+row["status"]] += 1
        if "metrics" in row:
            groups[unit].append(row)
        if row["status"]=="outside" and "max_score" in row:
            outside[unit].append(row)
    summary = {}
    for key,vals in groups.items():
        is_light = vals[0]["truth_unit"]=="device-owner-proxy"
        def means(chunk):
            return {k:float(np.mean([r["metrics"][k] for r in chunk])) for k in chunk[0]["metrics"]}
        def reduce_group(records):
            if not is_light:
                return means(records)
            experiments = defaultdict(list)
            for r in records:
                experiments[r["recording"]].append(r)
            return means([{"metrics":means(v)} for v in experiments.values()])
        def stratum(records):
            return {"events":len(records),"metrics":reduce_group(records)}
        summary[key] = stratum(vals)
        summary[key]["root_count"] = {kind:stratum(v) for kind in ("single","multiple")
                                      if (v:=[r for r in vals if ("single" if len(r["targets"])==1 else "multiple")==kind])}
        units = defaultdict(list)
        for r in vals:
            units[r["group"]].append(r)
        names = sorted(units)
        summary[key]["independent_groups"] = len(names)
        if len(names)>1:
            rng = np.random.default_rng(20261126)
            # Each resampled cluster includes all its incidents. Preserve repeated experiment draws.
            keys = list(vals[0]["metrics"])
            weights = np.array([1 if is_light else len(units[n]) for n in names])
            group_values = np.array([[means(units[n])[k] for k in keys] for n in names])
            indices = rng.integers(len(names),size=(10000,len(names)))
            boot = (group_values[indices]*weights[indices,None]).sum(1)/weights[indices].sum(1)[:,None]
            summary[key]["cluster_95ci"] = {k:np.quantile(boot[:,j],[.025,.975]).tolist() for j,k in enumerate(keys)}
    outside_summary = {}
    for key,records in outside.items():
        scores = np.array([r["max_score"] for r in records])
        outside_summary[key] = {"n":len(scores),"max_score_quantiles":np.quantile(scores,[0,.25,.5,.75,1]).tolist()}
        if all("max_probability" in r for r in records):
            outside_summary[key]["asserted_fraction"] = float(np.mean(scores>=.5))
    return {"ranking":summary,"status_counts":dict(counts),"outside":outside_summary}


def score_record(model,base,rec,threshold):
    path = base/rec["file"]
    if digest(path)!=rec["sha256"]:
        raise ValueError(f"Input checksum mismatch: {path}")
    with np.load(path,allow_pickle=False) as f:
        x,t,names,ref = f["x"],f["time"],f["columns"].tolist(),f["reference"]
    if names != rec["columns"]:
        raise ValueError("Recording column order mismatch")
    if not len(ref):
        ref = None
    candidates = rec["candidate_ids"]
    members = rec.get("entity_members",{v:[v] for v in candidates})
    if set(members)!=set(candidates) or len(candidates)!=len(set(candidates)):
        raise ValueError("Candidate/member mismatch")
    ix = [[names.index(s) for s in members[c]] for c in candidates]
    if any(not a for a in ix):
        raise ValueError("Every ranked entity must contain a measured channel")
    events = rec["events"]
    if len(set(e["id"] for e in events))!=len(events):
        raise ValueError("Duplicate event IDs")
    if any(not 0<=e["start"]<e["stop"]<=len(t) for e in events):
        raise ValueError("Event bounds outside retained rows")
    cache = {}
    def infer(span):
        if span not in cache:
            cache[span] = model.root(x,t,*span,ref=ref)
        return cache[span]
    def make_row(e,span,mode):
        row = dict(dataset=rec["dataset"],release=rec["release"],truth_unit=rec["truth_unit"],
                   recording=rec["id"],group=rec["group"],event=e["id"],mode=mode,status=e["status"],
                   start=e["start"],stop=e["stop"],targets=e["targets"],candidates=candidates,
                   evidence_ref=e.get("evidence_ref",rec.get("source_url")),matched=span is not None)
        if e["status"] not in ("eligible","outside"):
            return row
        if e["status"]=="eligible" and (not e["targets"] or not set(e["targets"]).issubset(candidates)):
            raise ValueError("Eligible event needs documented targets inside the panel")
        if span is None:
            scores = np.zeros(len(candidates))
            means = scores.copy()
        else:
            prob, evidence, windows = infer(span)
            scores = np.array([prob[j].max() for j in ix])
            means = np.array([prob[j].mean() for j in ix])
            row.update(scored_span=list(span),windows=windows,scores=scores.tolist(),
                       evidence=[bool(evidence[j].any()) for j in ix],max_score=float(scores.max()))
            if not isinstance(model,Severity):
                row["max_probability"] = row["max_score"]
        if e["status"]=="outside":
            return row
        if len(set(e["targets"]))==len(candidates):
            row["status"] = "all-positive"
            return row
        met = ranking(candidates,scores,e["targets"],rec["release"],rec["id"])
        row["metrics"] = met if span is not None else {k:0. for k in met}
        if "entity_members" in rec and span is not None:
            row["mean_entity_metrics"] = ranking(candidates,means,e["targets"],rec["release"],rec["id"])
        return row
    rows = [make_row(e,(e["start"],e["stop"]),"given-event") for e in events]
    alarms = None
    if threshold is not None and rec.get("alarm_evaluable",True):
        score, evidence = model.anomaly(x,t,ref=ref)
        preds = segments((score>=threshold)&(evidence>0))
        matching = match_events(events,preds)
        rows.extend(make_row(e,preds[matching[i]] if i in matching else None,"detected-event")
                    for i,e in enumerate(events))
        tp,fp = len(matching),len(preds)-len(matching)
        seconds = rec.get("seconds_per_clock_unit")
        alarms = {"recording":rec["id"],"dataset":rec["dataset"],"events":len(events),"alerts":len(preds),
                  "matched":tp,"precision":tp/len(preds) if preds else None,
                  "recall":tp/len(events) if events else None,"alerts_without_match":fp,
                  "false_per_1000_rows":1000*fp/len(t),"threshold":threshold,
                  "false_per_hour":3600*fp/((t[-1]-t[0])*seconds) if seconds and t[-1]>t[0] else None,
                  "no_evidence_rows":int((evidence==0).sum()),"intervals":preds}
    return rows,alarms


def run(args):
    if args.threshold is not None and not 0<=args.threshold<=1:
        raise ValueError("Threshold must come from calibrated source probabilities")
    ledger = json.loads(args.ledger.read_text(encoding="utf8"))
    if not ledger:
        raise ValueError("Empty release ledger")
    if getattr(args,"predictor","shifted")=="severity":
        model = Severity()
        identity = model.identity
    else:
        model = Predictor(args.checkpoint,args.device)
        identity = {"predictor":"shifted","checkpoint_sha256":digest(args.checkpoint)}
    rows, alarms = [],[]
    for rec in ledger:
        scores, detection = score_record(model,args.data,rec,args.threshold)
        rows.extend(scores)
        if detection is not None:
            alarms.append(detection)
    save_json(args.output/"events.json",rows)
    save_json(args.output/"summary.json",{**identity,
              "ledger_sha256":digest(args.ledger),"threshold":args.threshold,
              "alarms":alarms,**aggregate(rows)})
