"""Exact synthetic validation, held-out calibration, and test reporting (E.1).

Each rank writes bounded-size batches of tied score counts to a private SQLite
file on shared storage. Rank 0 combines equal float32 probabilities before
evaluating AP. Memory does not grow with the number of known time points.
"""
import csv
import gzip
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .data import FAMILIES, MODES, atomic_json, collate
from .point_tallies import AncestorWriter, ancestor_code, publish_tallies


SCHEMA = """
CREATE TABLE scores (
 task INTEGER, probability REAL, width INTEGER, mode INTEGER, total INTEGER, positives INTEGER,
 PRIMARY KEY (task,probability,width,mode)
) WITHOUT ROWID;
CREATE TABLE windows (
 idx INTEGER PRIMARY KEY, window_id TEXT, width INTEGER, sources INTEGER, mode INTEGER,
 family INTEGER, ancestor INTEGER, known_roots INTEGER, positive_roots INTEGER,
 root_unseen INTEGER, unseen_positive_roots INTEGER, eligible_roots INTEGER,
 real_rows INTEGER, known_rows INTEGER, positive_rows INTEGER, point_unseen INTEGER,
 root_ap REAL, healthy_max REAL
);
"""
ADD_SCORES = """INSERT INTO scores VALUES (?,?,?,?,?,?)
 ON CONFLICT (task,probability,width,mode) DO UPDATE SET
 total=total+excluded.total, positives=positives+excluded.positives"""
WINDOW_COLUMNS = ("idx","window_id","width","sources","mode","family","ancestor","known_roots",
                  "positive_roots","root_unseen","unseen_positive_roots","eligible_roots",
                  "real_rows","known_rows","positive_rows","point_unseen","root_ap","healthy_max")


def average_precision(scores, positive):
    """Tie-grouped AP; empty sets or sets with no positives have no AP."""
    if not np.any(positive):
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    endpoints = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    tp = np.cumsum(positive[order], dtype=np.int64)[endpoints]
    return float((np.diff(tp,prepend=0)*(tp/(1+endpoints))).sum() / tp[-1])


class TiedScores:
    def __init__(self, db, task):
        self.db = db
        self.task = task
        self.codes = []
        self.labels = []
        self.size = 0

    def add(self, score, labels, width, mode):
        if not len(score):
            return
        if not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
            raise ValueError("Non-finite or out-of-range probability.")
        # Exact float32 bit pattern; the low bytes store the two stratum keys.
        code = (score.astype(np.float32,copy=False).view(np.uint32).astype(np.uint64) << 16)
        code |= width*256+mode
        self.codes.append(code)
        self.labels.append(labels)
        self.size += len(code)
        if self.size >= 512_000:
            self.flush()

    def flush(self):
        if not self.size:
            return
        unique, inverse, counts = np.unique(np.concatenate(self.codes), return_inverse=True,
                                            return_counts=True)
        positive = np.bincount(inverse,weights=np.concatenate(self.labels),minlength=len(unique))
        prob = (unique >> 16).astype(np.uint32).view(np.float32)
        width, mode = (unique >> 8) & 255, unique & 255
        self.db.executemany(ADD_SCORES, ((self.task,float(s),int(n),int(m),int(k),int(p))
                             for s,n,m,k,p in zip(prob,width,mode,counts,positive)))
        self.codes.clear()
        self.labels.clear()
        self.size = 0


def _write_rank(model, dataset, indices, path, device, batch_size, workers, save_points=False):
    device = torch.device(device)
    loader = DataLoader(Subset(dataset,indices),batch_size=batch_size,shuffle=False,
                        collate_fn=collate,num_workers=workers,pin_memory=device.type=="cuda")
    db = sqlite3.connect(path)
    writer = AncestorWriter(path.with_suffix(".points")) if save_points else None
    try:
        db.executescript(SCHEMA)
        root_hist, point_hist = TiedScores(db,0), TiedScores(db,1)
        placeholders = ",".join("?" for _ in WINDOW_COLUMNS)
        for batch in loader:
            features = [batch[name].to(device,non_blocking=True) for name in
                        ("values","observed","time","channel_valid","row_valid")]
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                       enabled=device.type=="cuda"):
                out = model(*features)
            def get_score(logits,valid):
                score = torch.where(out[valid],out[logits].float().sigmoid(),0.0)
                return score.cpu().numpy(),out[valid].cpu().numpy()
            root, root_seen = get_score("root_logits","root_valid")
            point, point_seen = get_score("anomaly_logits","anomaly_valid")
            arrays = {name:batch[name].numpy() for name in
                      ("index","channel_valid","E","R","R_known","row_valid","Y","Y_known")}
            rows = []
            for j,idx in enumerate(arrays["index"]):
                rec = dataset.table[idx]
                n,k,m,fa,anc = (int(rec[f]) for f in ("n","k","mode","family","ancestor"))
                eligible = arrays["channel_valid"][j] & arrays["E"][j]
                r_known = eligible & arrays["R_known"][j]
                actual_rows = arrays["row_valid"][j]
                y_known = actual_rows & arrays["Y_known"][j]
                rs, ry = root[j,r_known], arrays["R"][j,r_known]
                ps, py = point[j,y_known], arrays["Y"][j,y_known]
                root_hist.add(rs,ry,n,m)
                point_hist.add(ps,py,n,m)
                if writer is not None:
                    writer.add(ancestor_code(n,fa,anc),ps,py)
                nr, np_root = len(ry), int(ry.sum())
                ap = average_precision(rs,ry) if m != 0 and 0 < np_root < nr else None
                # Unknown rows cannot establish a known-healthy calibration window.
                max_healthy = None
                if m == 0 and (y_known == actual_rows).all() and len(py) and not py.any():
                    max_healthy = float(ps.max())
                w_id = bytes(rec["id"]).decode("ascii")
                rows.append((int(idx),w_id,n,k,m,fa,anc,nr,np_root,
                             int((r_known & ~root_seen[j]).sum()),
                             int((r_known & arrays["R"][j] & ~root_seen[j]).sum()),int(eligible.sum()),
                             int(actual_rows.sum()),len(py),int(py.sum()),
                             int((y_known & ~point_seen[j]).sum()),ap,max_healthy))
            db.executemany(f"INSERT INTO windows VALUES ({placeholders})",rows)
        root_hist.flush()
        point_hist.flush()
        db.commit()
    finally:
        db.close()
        if writer is not None:
            writer.close()


def _combine(db, path):
    db.execute("ATTACH DATABASE ? AS part",(str(path),))
    try:
        db.execute("INSERT INTO windows SELECT * FROM part.windows")
        db.execute("""INSERT INTO scores SELECT * FROM part.scores WHERE 1
         ON CONFLICT (task,probability,width,mode) DO UPDATE SET
         total=total+excluded.total,positives=positives+excluded.positives""")
        db.commit()
    finally:
        db.execute("DETACH DATABASE part")


def _score_summary(db, task, field=None, value=None, calibration=False):
    if field not in {None,"width","mode"}:
        raise ValueError("Unknown score stratum")
    condition = "task=?" if field is None else f"task=? AND {field}=?"
    args = (task,) if field is None else (task,value)
    cursor = db.execute(f"""SELECT probability,SUM(total),SUM(positives) FROM scores
              WHERE {condition} GROUP BY probability ORDER BY probability DESC""",args)
    total, hits = 0,0
    ap_numerator, squared_error = 0.0,0.0
    bins = np.zeros((15,3),dtype=np.float64)
    while values := cursor.fetchmany(16384):
        a = np.asarray(values,dtype=np.float64)
        s,n,p = a.T
        rank = total + np.cumsum(n)
        found = hits + np.cumsum(p)
        ap_numerator += (p*found/rank).sum()
        total, hits = int(rank[-1]), int(found[-1])
        if calibration:
            squared_error += (p*(1-s)**2 + (n-p)*s**2).sum()
            which = np.minimum((s*15).astype(int),14)
            for col,weights in enumerate((n,p,n*s)):
                bins[:,col] += np.bincount(which,weights=weights,minlength=15)
    summary = dict(known=total,positive=hits,ap=float(ap_numerator/hits) if hits else None)
    if calibration:
        ece = np.abs(bins[:,1]-bins[:,2]).sum()
        summary.update(brier=float(squared_error/total) if total else None,
                       ece=float(ece/total) if total else None,
                       bins=[dict(lower=i/15,upper=(i+1)/15,count=int(n),positive=int(p),
                                  positive_rate=float(p/n) if n else None,
                                  mean_probability=float(q/n) if n else None)
                             for i,(n,p,q) in enumerate(bins)])
    return summary


def _window_summary(db, field=None):
    fields = {None:("1","all"),"width":("width","width"),"sources":("sources","sources"),
              "mode":("mode","mode"),"family":("family","family")}
    group, _ = fields[field]
    # Use a literal string for the pooled grouping; GROUP BY 1 denotes a SELECT index.
    if field is None:
        group = "'all'"
    query = f"""SELECT {group},COUNT(*),COUNT(root_ap),AVG(root_ap),
        SUM(mode=0),SUM(known_roots=0),SUM(known_roots>0 AND positive_roots=0),
        SUM(known_roots>0 AND positive_roots=known_roots),SUM(eligible_roots),
        SUM(known_roots),SUM(root_unseen),SUM(unseen_positive_roots),SUM(real_rows),SUM(known_rows),
        SUM(point_unseen),COUNT(healthy_max) FROM windows GROUP BY {group}"""
    names = ("windows","rankable_windows","root_map","healthy_windows","all_unknown_root_windows",
             "all_negative_root_windows","all_positive_root_windows","eligible_roots",
             "known_roots","known_roots_without_evidence","positive_roots_without_evidence",
             "real_rows","known_rows","known_rows_without_evidence","known_healthy_windows")
    result = {}
    for key,*row in db.execute(query):
        if field == "mode":
            key = MODES[key]
        elif field == "family":
            key = FAMILIES[key]
        result[str(key)] = dict(zip(names,row))
    return result


def _save_event_rows(db, path):
    temp = path.with_suffix(".part")
    with gzip.open(temp,"wt",encoding="utf8",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(WINDOW_COLUMNS)
        for row in db.execute("SELECT * FROM windows ORDER BY idx"):
            fields = list(row)
            fields[4],fields[5] = MODES[fields[4]],FAMILIES[fields[5]]
            writer.writerow(fields)
    temp.replace(path)


def _summarise(db, output, purpose, dataset, parts):
    root = _score_summary(db,0,calibration=True)
    point = _score_summary(db,1,calibration=True)
    summary = _window_summary(db)["all"]
    point_metrics = {}
    for field,key in (("width","by_width"),("mode","by_mode")):
        scores = {}
        for (v,) in db.execute(f"SELECT DISTINCT {field} FROM windows ORDER BY {field}"):
            label = MODES[v] if field=="mode" else str(v)
            scores[label] = _score_summary(db,1,field,v)
        point_metrics[key] = scores
    m, ap = summary["root_map"],point["ap"]
    report = dict(purpose=purpose,selection_score=(m+ap)/2 if m is not None and ap is not None else None,
                  root_map=m,point_ap=ap,threshold=None,root_assertion_threshold=.5,
                  denominators=summary,root_pooled=root,point_pooled=point,
                  window_metrics={key:_window_summary(db,col) for key,col in
                                  (("by_width","width"),("by_K","sources"),("by_mode","mode"),
                                   ("by_family","family"))},point_metrics=point_metrics)
    if purpose == "calibration":
        peaks = np.fromiter((r[0] for r in db.execute(
            "SELECT healthy_max FROM windows WHERE healthy_max IS NOT NULL")),dtype=float)
        if len(peaks):
            report["threshold"] = float(np.quantile(peaks,.95,method="linear"))
        report["threshold_windows"] = len(peaks)
    if purpose == "test":
        path = output.with_name(output.stem+".windows.csv.gz")
        _save_event_rows(db,path)
        report["event_records"] = path.name
        point_path = output.with_name(output.stem+".ancestor-points.npy")
        publish_tallies(parts,point_path)
        report["ancestor_points"] = point_path.name
        report["manifest_sha256"] = getattr(dataset,"meta",{}).get("manifest_sha256")
    atomic_json(output,report)
    return report


def validate(model, dataset, indices, output: Path, device: torch.device, batch_size: int,
             workers: int, purpose="selection") -> dict:
    """Called by every rank with the same global indices and shared output path."""
    if purpose not in {"selection","calibration","test"}:
        raise ValueError("Unknown validation purpose")
    output = Path(output)
    if output.suffix.lower() != ".json":
        output = output/f"{purpose}.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or np.min(indices)<0 or np.max(indices)>=len(dataset) or len(np.unique(indices))!=len(indices)):
        raise ValueError("Evaluation indices must be distinct, nonempty corpus coordinates")
    distributed = dist.is_available() and dist.is_initialized()
    rank,world = (dist.get_rank(),dist.get_world_size()) if distributed else (0,1)
    # Globally group widths, then take disjoint unequally sized rank partitions
    # without DistributedSampler's tail duplication.
    if purpose == "test":
        table = dataset.table[indices]
        order = indices[np.lexsort((table["ancestor"],table["family"],table["n"]))]
    else:
        order = indices[np.argsort(dataset.table["n"][indices],kind="stable")]
    work = [tempfile.mkdtemp(prefix=".scores-",dir=output.parent) if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(work,src=0)
    tmp = Path(work[0])
    was_training = model.training
    model.eval()
    error = None
    try:
        _write_rank(model,dataset,order[rank::world],tmp/f"rank-{rank}.db",device,batch_size,workers,purpose=="test")
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    finally:
        model.train(was_training)
    errors = [error]
    if distributed:
        errors = [None]*world
        dist.all_gather_object(errors,error)
    if any(errors):
        raise RuntimeError(f"Validation failed; retained {tmp}: " + " | ".join(filter(None,errors)))
    outcome = [None,None]
    if rank == 0:
        db = sqlite3.connect(tmp/"rank-0.db")
        try:
            for child in range(1,world):
                _combine(db,tmp/f"rank-{child}.db")
            if db.execute("SELECT COUNT(*) FROM windows").fetchone()[0] != len(indices):
                raise ValueError("Validation window coverage mismatch")
            outcome[1] = _summarise(db,output,purpose,dataset,(tmp/f"rank-{r}.points" for r in range(world)))
        except Exception as exc:
            outcome[0] = f"{type(exc).__name__}: {exc} (retained {tmp})"
        finally:
            db.close()
        if outcome[0] is None:
            shutil.rmtree(tmp)
    if distributed:
        dist.broadcast_object_list(outcome,src=0)
    if outcome[0]:
        raise RuntimeError(outcome[0])
    return outcome[1]
