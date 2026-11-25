"""Materialize inputs and frozen event metadata. All event bounds are half-open row indices."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .assets import AERCA_REV, CHAMBERS, digest, download, git_blob, save_json

LIGHT = ["current","angle_1","angle_2","ir_1","vis_1","ir_2","vis_2","ir_3","vis_3"]
WIND = ["current_in","current_out","rpm_in","rpm_out","pressure_upwind","pressure_downwind",
        "pressure_ambient","pressure_intake","mic","signal_1","signal_2"]


def segments(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask,dtype=bool), False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def recording(base, name, x, cols, time, ref=None):
    folder = base/"rca"
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/(name+".npz")
    x = np.asarray(x, dtype="float32")
    t = np.asarray(time, dtype="float64")
    if len(set(cols)) != len(cols) or x.ndim != 2 or x.shape != (len(t),len(cols)):
        raise ValueError(f"Invalid feature panel for {name}")
    if len(t) == 0 or not np.isfinite(t).all() or (np.diff(t)<=0).any():
        raise ValueError(f"Clock must be finite, nonempty and increasing in {name}")
    if ref is not None and (ref.ndim != 2 or ref.shape[1] != len(cols)):
        raise ValueError("Reference channel order/width mismatch")
    np.savez_compressed(path, x=x, time=t, columns=np.asarray(cols,dtype=str),
                        reference=ref if ref is not None else np.empty((0,len(cols)),np.float32))
    return {"file":str(path.relative_to(base)), "sha256":digest(path), "columns":cols}


def chamber(base):
    records = []
    for release, columns in zip(CHAMBERS,(LIGHT,WIND)):
        folder = base/release
        light = columns == LIGHT
        ref = pd.read_csv(folder/"uniform_reference.csv")[columns].to_numpy("float32") if light else None
        if light:
            names = {f"uniform_v_{param}_{strength}": target for param,target in
                     [("angle_1","angle_1"),("angle_2","angle_2"),("c","current")]
                     for strength in ("mid","strong")}
        else:
            names = {p.stem:None for p in sorted(folder.glob("*.csv"))
                     if "load_in" in pd.read_csv(p,nrows=0).columns}
        if not names:
            raise ValueError("No eligible Causal Chambers experiments found")
        for name,target in names.items():
            src = folder/(name+".csv")
            df = pd.read_csv(src)
            t = df["timestamp"].to_numpy(dtype=float)
            rec = recording(base, release+"_"+name, df[columns],columns,t,ref)
            rec.update(dataset="CausalChambers",release=release,id=name,group=name,
                       truth_unit="device-owner-proxy" if light else "outside-measurement-panel",
                       source_sha256=digest(src),source_url="https://github.com/juangamella/causal-chamber/tree/main/datasets/"+release,
                       seconds_per_clock_unit=1.0, candidate_ids=columns, alarm_evaluable=False)
            spans = [(i,min(len(df),i+512)) for i in range(0,max(1,len(df)-511),512)] if light else [(0,len(df))]
            rec["events"] = [dict(id=f"{name}-{a}",start=a,stop=b,targets=[target] if light else [],
                                  status="eligible" if light else "outside") for a,b in spans]
            records.append(rec)
    save_json(base/"rca"/"chamber.json", records)


def msds(base):
    archive_url = "https://zenodo.org/api/records/3549604/files/concurrent%20data.zip/content"
    archive = download(archive_url,base/"archives"/"msds-concurrent.zip",
                       "be2ba2f17f774e7f3f045dcf5f406ff0","md5")
    labels_url = f"https://raw.githubusercontent.com/hanxiao0607/AERCA/{AERCA_REV}/datasets/msds/labels.csv"
    label_path = download(labels_url,base/"msds"/"labels.csv")
    if git_blob(label_path.read_bytes()) != "be5b52a6d4c4f169267ff30b3bb83f8b50719cc9":
        raise ValueError("Label file differs from the identical AERCA and TranAD release blobs")
    raw_frames = []
    hosts = [122,113,123,117,124]  # order used by the AERCA loader
    with zipfile.ZipFile(archive) as z:
        for host in hosts:
            suffix = f"wally{host}_metrics_concurrent.csv"
            names = [n for n in z.namelist() if n.endswith("/metrics/"+suffix) and not n.startswith("__MACOSX/")]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one MSDS metric file for {host}: {names}")
            with z.open(names[0]) as f:
                df = pd.read_csv(f).drop(columns=["load.cpucore","load.min1","load.min5","load.min15"])
            raw_frames.append(df)
    begin = max(df["now"].min() for df in raw_frames)
    end = min(df["now"].max() for df in raw_frames)
    frames = []
    for i,df in enumerate(raw_frames):
        # Both releases use index-label drop with positions from the already filtered table.
        # Preserving that historical quirk is necessary to recover the labelled clock.
        kept = df.drop(np.flatnonzero(df["now"] < begin))
        kept = kept.drop(np.flatnonzero(kept["now"] > end))
        frame = kept.melt(id_vars="now").dropna().pivot_table(index="now",columns="variable",values="value")
        frames.append(frame.add_suffix("_"+str(i)))
    panel = pd.concat(frames,axis=1,join="inner").sort_index()
    panel = panel.iloc[round(len(panel)*0.1):]
    split = round(len(panel)/2)
    train, test = panel.iloc[:split],panel.iloc[split:]
    raw = pd.read_csv(label_path)
    n_test_before_sampling = len(test)
    train, test = train.iloc[::5],test.iloc[::5]
    if (list(raw.columns[1:]) != list(map(str,range(10))) or test.shape != (len(raw),10)
            or not np.array_equal(raw.iloc[:,0].to_numpy(),np.arange(len(test)))):
        raise ValueError(f"AERCA alignment mismatch: metrics {test.shape}, labels {raw.shape}")
    # TranAD's original preprocess.py takes metrics [::5] and its published labels [::1].
    labels = raw.iloc[:,1:].to_numpy()
    if not np.isin(labels,[0,1]).all():
        raise ValueError("AERCA labels must be binary")
    cols = test.columns.tolist()
    # Zenodo uses a trailing 5-character timezone abbreviation. Relative seconds suffice.
    time = pd.DatetimeIndex(pd.to_datetime(test.index.str[:-5],format="%Y-%m-%d %H:%M:%S"))
    t = (time-time[0]).total_seconds().to_numpy()
    rec = recording(base, "msds-aerca", test,cols,t,train.to_numpy("float32"))
    rec.update(dataset="MSDS",id="concurrent",group="concurrent",release="Zenodo 3549604 v1.1 + AERCA "+AERCA_REV,
               source_url=archive_url, label_url=labels_url,label_sha256=digest(label_path),
               before_downsample=n_test_before_sampling,aligned_label_rows=len(test),
               metric_stride=5,label_stride=1,
               candidate_ids=cols,seconds_per_clock_unit=1.,truth_unit="metric-annotation-proxy",alarm_evaluable=True)
    rec["events"] = [dict(id=f"episode-{a}", start=a,stop=b,status="eligible",
                          targets=np.asarray(cols)[labels[a:b].any(0)].tolist())
                     for a,b in segments(labels.any(1))]
    save_json(base/"rca"/"msds.json",[rec])


def industrial_import(spec_path, base):
    """Import a licensed release using an audited column dictionary and same-release event table."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    home = spec_path.parent
    dataset = spec["dataset"]
    if dataset not in ("SWaT","WADI","MSDS"):
        raise ValueError("Expected SWaT, WADI or MSDS")
    def read(src):
        opts = src.get("read_options",{})
        file = home/src["path"]
        return pd.read_excel(file,**opts) if file.suffix.lower() in (".xls",".xlsx") else pd.read_csv(file,**opts)
    cols = list(spec["features"])
    def features(df):
        return df[list(spec["features"].values())].apply(pd.to_numeric,errors="raise").to_numpy("float32")
    src = spec["recording"]
    raw = read(src)
    clock_str = raw[src["time_columns"]].astype(str).agg(" ".join,axis=1).str.strip()
    t = pd.DatetimeIndex(pd.to_datetime(clock_str,format=src["time_format"],errors="raise"))
    if t.tz is None:
        t = t.tz_localize(src["timezone"],ambiguous="raise",nonexistent="raise")
    # Absolute epoch seconds for event lookup, preserve source-row -> grouped-row map.
    sec = (t.tz_convert("UTC")-pd.Timestamp("1970-01-01",tz="UTC")).total_seconds().to_numpy()
    frame = pd.DataFrame(features(raw),columns=cols,index=sec)
    merged = frame.groupby(level=0,sort=True).median()
    time = merged.index.to_numpy()
    ref = features(read(spec["reference"])) if spec.get("reference") else None
    rec = recording(base,spec["id"],merged,cols,time,ref)
    rec.update({k:spec[k] for k in ("dataset","id","group","release","truth_unit","candidate_ids","source_url")})
    rec.update(seconds_per_clock_unit=1.,alarm_evaluable=True,raw_sha256=digest(home/src["path"]),
               reference_sha256=digest(home/spec["reference"]["path"]) if ref is not None else None)
    if "entity_members" in spec:
        rec["entity_members"] = spec["entity_members"]
    ledger = home/spec["events_json"]
    events = json.loads(ledger.read_text(encoding="utf8"))
    rec["event_sha256"] = digest(ledger)
    for e in events:
        if not e.get("evidence_ref") or not set(e["targets"]).issubset(rec["candidate_ids"]) and e["status"]=="eligible":
            raise ValueError(f"Unverified target for {e['id']}")
        def locate(field,side):
            v = pd.Timestamp(e[field])
            if v.tz is None:
                v = v.tz_localize(src["timezone"],ambiguous="raise",nonexistent="raise")
            return int(np.searchsorted(time,v.timestamp(),side=side))
        e["start"] = locate("start_time","left")
        e["stop"] = locate("end_time","right" if e.get("end_inclusive",True) else "left")
        if not 0 <= e["start"] < e["stop"] <= len(time):
            raise ValueError(f"Event outside recording or empty: {e['id']}")
    rec["events"] = events
    folder = base/"rca"
    row_map = np.searchsorted(time,sec)
    np.save(folder/(spec["id"]+"-original-row-to-group.npy"),row_map)
    if "label_column" in src:
        values = raw[src["label_column"]].astype(str).str.strip().str.lower()
        mapping = {str(k).strip().lower():int(v) for k,v in src["label_mapping"].items()}
        if dataset == "WADI" and (mapping.get("-1")!=1 or mapping.get("1")!=0):
            raise ValueError("WADI.A2 requires -1=attack and +1=normal (mapping keys -1 and 1)")
        labels = values.map(mapping)
        if labels.isna().any():
            raise ValueError("Unrecognized release label")
        native = np.zeros(len(time),dtype=bool)
        np.logical_or.at(native,row_map,labels.to_numpy(dtype=bool))
        registered = np.zeros_like(native)
        for e in events:
            registered[e["start"]:e["stop"]] = True
        rec["label_disagreement_rows"] = int(np.sum(registered!=native))
        np.save(folder/(spec["id"]+"-native-point-labels.npy"),native)
    save_json(folder/(spec["id"]+".json"),[rec])
