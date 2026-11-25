"""Pinned paper and TSB-AD protocols; preserve one score per scored row."""
import importlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import TIMERCD_REV, TSB_REV, digest, save_json, metric_source
from .inference import Predictor
from .baselines import AuthorTimeRCD, Severity
from .adapters import segments

PAPER = {
    "U":{"UCR":228,"NAB":28,"YAHOO":259,"IOPS":17,"MGAB":9,"SED":3,"Stock":20,
         "TODS":15,"NEK":9,"Power":1,"WSD":111},
    "M":{"MSL":16,"PSM":1,"SMAP":27,"SMD":22,"SWaT":2},
}
TIME_METRICS = ("VUS-PR","VUS-ROC","Affiliation-F","Affiliation-P","Affiliation-R","Standard-F1",
                "Standard-Precision","Standard-Recall","F1_T","Precision_T","Recall_T","AUC-PR",
                "AUC-ROC","PA-F1","PA-Precision","PA-Recall")
TSB_METRICS = ("AUC-PR","AUC-ROC","VUS-PR","VUS-ROC","Standard-F1","PA-F1",
               "Event-based-F1","R-based-F1","Affiliation-F")


def file_info(name):
    match = re.fullmatch(r"\d+_(.+?)_id_.+_tr_(\d+)_1st_[^.]+\.csv", name)
    if match is None:
        raise ValueError(f"Invalid TSB-AD filename: {name}")
    return match[1], int(match[2])


def inventory(base, split, paper):
    folder = base/("TSB-AD-"+split)
    if paper:
        files = sorted(p for p in folder.glob("*.csv") if file_info(p.name)[0] in PAPER[split])
        found = Counter(file_info(p.name)[0] for p in files)
        if found != PAPER[split]:
            raise ValueError(f"TimeRCD table counts mismatch ({split}): actual {dict(found)}, expected {PAPER[split]}")
    else:
        lists = metric_source(base,"TSB")/"Datasets"/"File_List"
        def names(key):
            return pd.read_csv(lists/f"TSB-AD-{split}-{key}.csv")["file_name"].tolist()
        eva,tune = names("Eva"),names("Tuning")
        count = 350 if split=="U" else 180
        if len(set(eva))!=count or len(eva)!=count or set(eva)&set(tune):
            raise ValueError("Invalid official evaluation split")
        files = [folder/n for n in eva]
    if not all(f.is_file() for f in files):
        raise FileNotFoundError("Missing official series. Run prepare --assets tsad.")
    return files


def backend(base, paper, engine="fast"):
    path = metric_source(base,"TimeRCD" if paper else "TSB")
    sys.path.insert(0,str(path.resolve()))
    if not paper:
        metric = importlib.import_module("TSB_AD.evaluation.metrics").get_metrics
        period = importlib.import_module("TSB_AD.utils.slidingWindows").find_length_rank
        return period, lambda y,s,w: metric(s,y,slidingWindow=w)
    # Use underlying author functions directly: the public wrapper silently returns 0 on some exceptions.
    metrics = importlib.import_module("evaluation.basic_metrics")
    period = importlib.import_module("utils.slidingWindows").find_length_rank
    def compute(y,s,w):
        grader = metrics.basic_metricor()
        values = metrics.generate_curve(y,s,w,version="opt",thre=250)
        if engine=="fast":
            # Serial order for the affiliation grid makes a tied maximum deterministic.
            # Four-worker chunked routines use the same grids as the authors' fast runner.
            grid = grader._compute_affiliation_chunk(np.linspace(0.8,1,300),s,y)
            best = max(grid,key=lambda item:item["f1"])
            affiliation = (best["f1"],best["precision"],best["recall"])
            f1 = grader.metric_standard_F1_chunked(y,s,chunk_size=25,num_workers=4)
            time_f1 = grader.metric_F1_T(y,s,chunk_size=25,max_workers=4,n_splits=800)
            pa = grader.metric_PointF1PA_chunked(y,s,chunk_size=30,num_workers=4)
        else:
            affiliation = grader.metric_Affiliation(y,s)
            f1 = grader.metric_standard_F1(y,s)
            time_f1 = grader.metric_F1_T(y,s)
            pa = grader.metric_PointF1PA(y,s)
        return {"VUS-PR":values[-1],"VUS-ROC":values[-2],"Affiliation-F":affiliation[0],
                "Affiliation-P":affiliation[1],"Affiliation-R":affiliation[2],
                "Standard-F1":f1["F1"],"Standard-Precision":f1["Precision"],"Standard-Recall":f1["Recall"],
                "F1_T":time_f1["F1_T"],"Precision_T":time_f1["P_T"],"Recall_T":time_f1["R_T"],
                "AUC-PR":grader.metric_PR(y,s),"AUC-ROC":grader.metric_ROC(y,s),
                "PA-F1":pa["F1_PA"],"PA-Precision":pa["P_PA"],"PA-Recall":pa["R_PA"]}
    return period, compute


def summarize(rows):
    def means(records):
        keys = records[0]["metrics"]
        return reduce_metrics(records,keys,"files")
    def reduce_metrics(records,keys,unit):
        by_metric = {k:[r["metrics"][k] for r in records if r["metrics"].get(k) is not None] for k in keys}
        return {unit:len(records),"finite_denominator":{k:len(v) for k,v in by_metric.items()},
                "metrics":{k:float(np.mean(v)) if v else None for k,v in by_metric.items()}}
    by_dataset = defaultdict(list)
    by_split = defaultdict(list)
    for row in rows:
        by_dataset[(row["split"],row["dataset"])].append(row)
        by_split[row["split"]].append(row)
    per_source = {s+"-"+n:means(v) for (s,n),v in by_dataset.items()}
    def dataset_mean(values):
        return reduce_metrics(values,values[0]["metrics"],"datasets")
    return {"dataset":per_source, "dataset_equal":dataset_mean(list(per_source.values())),
            "split_dataset_equal":{s:dataset_mean([v for k,v in per_source.items() if k.startswith(s+"-")])
                                   for s in by_split},
            "split_series_weighted":{s:means(v) for s,v in by_split.items()}}


def run(args):
    base, out = args.data, args.output
    paper = args.action == "timercd"
    predictor_name = getattr(args,"predictor","shifted")
    width = args.width or (5000 if paper else (15000 if predictor_name=="author" else 2048))
    if predictor_name=="author" and args.stride is not None:
        raise ValueError("Original TimeRCD uses its own nonoverlapping windows; omit --stride")
    stride = args.stride if args.stride is not None else (width if predictor_name=="author" else width//2)
    splits = ["U","M"] if args.split=="both" else [args.split]
    jobs = [(s,p) for s in splits for p in inventory(base,s,paper)]
    engine = getattr(args,"metric_engine","official")
    period, compute = backend(base,paper,engine)
    if predictor_name=="author":
        predictor = AuthorTimeRCD(base,args.device,paper,width)
        identity = predictor.identity
    elif predictor_name=="severity":
        predictor = Severity()
        identity = predictor.identity
    else:
        predictor = Predictor(args.checkpoint,args.device)
        identity = {"predictor":"shifted","checkpoint_sha256":digest(args.checkpoint)}
    key = {**identity,"protocol":args.action,
           "metric_commit":TIMERCD_REV if paper else TSB_REV,"metric_engine":engine,
           "width":width,"stride":stride,"protocol_version":5}
    rows = []
    for split, path in jobs:
        group, prefix = file_info(path.name)
        result_file = out/split/(path.stem+".json")
        score_file = result_file.with_suffix(".npz")
        descriptor = dict(key,source_sha256=digest(path))
        if result_file.exists() and score_file.exists():
            cached = json.loads(result_file.read_text(encoding="utf8"))
            if cached["identity"]==descriptor and cached["scores_sha256"]==digest(score_file):
                rows.append(cached)
                continue
        frame = pd.read_csv(path)
        if frame.columns[-1] != "Label":
            raise ValueError(f"Unexpected label column: {path}")
        original = frame
        frame = original.dropna()
        x = frame.iloc[:,:-1].to_numpy(dtype=np.float32)
        y = frame["Label"].to_numpy()
        if not np.isin(y,[0,1]).all() or not 0<=prefix<len(y):
            raise ValueError(f"Invalid label/prefix in {path}")
        start = prefix if paper else 0
        time = np.arange(len(y),dtype=np.float64)
        scores, hits = predictor.anomaly(x,time,first=start,ref=x[:prefix] if prefix else None,
                                         width=width,stride=stride)
        # The TimeRCD runner passes the entire 2-D data matrix to its period finder.
        # The official TSB-AD runner calls the first-channel period for both U and M.
        sw = int(period(x if paper else x[:,0:1],rank=1))
        label = y[start:].astype(int)
        result_file.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(score_file,score=scores,valid_windows=hits,
                            source_rows=frame.index.to_numpy()[start:],label=label)
        official_degenerate = None
        if np.unique(label).size < 2:
            # These real release files still receive scores. Event ranking needs both classes.
            # Report the exact public wrapper's convention/failure alongside the undefined panel.
            met = dict.fromkeys(TIME_METRICS if paper else TSB_METRICS)
            status = "all-positive" if label[0] else "all-negative"
            upstream = importlib.import_module("evaluation.metrics" if paper else "TSB_AD.evaluation.metrics")
            try:
                if paper and engine=="fast":
                    raw = upstream.get_metrics_optimized(scores,label,slidingWindow=sw,heavy_workers=2,light_workers=2)
                else:
                    raw = upstream.get_metrics(scores,label,slidingWindow=sw)
                official_degenerate = {"metrics":{k:float(v) if np.isfinite(v) else None for k,v in raw.items()}}
            except Exception as exc:
                official_degenerate = {"error":f"{type(exc).__name__}: {exc}"}
        else:
            raw = compute(label,scores,sw)
            met = {k:float(v) if np.isfinite(v) else None for k,v in raw.items()}
            status = "scored" if all(v is not None for v in met.values()) else "official-nonfinite"
        rec = {"identity":descriptor,"file":path.name,"split":split,"dataset":group,"start_row_retained":start,
               "n":len(scores),"positive_rows":int(label.sum()),"without_evidence":int((hits==0).sum()),
               "normalization":"authors-scored-input" if predictor_name=="author" else
                               ("designated-prefix" if prefix else "history-only"),"period":sw,
               "reference_positive_rows":int(y[:prefix].sum()),
               "original_rows":len(original),"retained_rows":len(frame),
               "dropped_label_counts":original.loc[~original.index.isin(frame.index),"Label"].value_counts(dropna=False).rename(index=str).to_dict(),
               "metrics":met,"metric_status":status,"official_degenerate":official_degenerate,
               "scores_sha256":digest(score_file)}
        if not paper:
            raw_y = original["Label"].to_numpy()
            known = np.isin(raw_y,[0,1])
            removed = ~original.index.isin(frame.index)
            rec["positive_events_with_removed_rows"] = sum(bool(removed[a:b].any()) for a,b in segments(raw_y==1))
            sensitivity = met
            if removed.any():
                raw_score = np.zeros(len(original))
                raw_score[frame.index.to_numpy()] = scores
                if np.unique(raw_y[known]).size==2:
                    z = compute(raw_y[known].astype(int),raw_score[known],sw)
                    sensitivity = {k:float(v) if np.isfinite(v) else None for k,v in z.items()}
                else:
                    sensitivity = dict.fromkeys(TSB_METRICS)
            rec["original_known_row_sensitivity"] = {"rows":int(known.sum()),"metrics":sensitivity}
        save_json(result_file,rec)
        rows.append(rec)
        print(f"[{len(rows)}/{len(jobs)}] {split}/{path.name} {status} VUS-PR={met['VUS-PR']}",flush=True)
    convention = []
    for r in rows:
        alternate = (r["official_degenerate"] or {}).get("metrics",r["metrics"])
        convention.append(dict(r,metrics=alternate))
    summary = {"identity":key,"splits":splits,**summarize(rows),
               "metric_status_counts":dict(Counter(r["metric_status"] for r in rows)),
               "official_degenerate_convention":summarize(convention)}
    if not paper:
        summary["original_known_row_sensitivity"] = summarize(
            [dict(r,metrics=r["original_known_row_sensitivity"]["metrics"]) for r in rows])
    save_json(out/"summary.json",summary)
