"""MSDS direct service-level injection ground truth from the public v1.1 concurrent release."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import ceil, floor
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from .adapters import industrial_import
from .assets import download, save_json

PREFIX = "concurrent data/"
LOG = PREFIX+"logs/logs_aggregated_concurrent.csv"
REF = "https://zenodo.org/records/3549604"
REPORTS = ("boot","image","network")


def iso(t):
    return datetime.fromtimestamp(t,timezone.utc).isoformat()


def observed(z,folder):
    # This function can run with only the one published service-log CSV.
    # No timestamps, durations, targets, texts, status or hooks from Rally are inputs.
    raw = pd.read_csv(z.open(LOG),usecols=["@timestamp","Logger"],low_memory=False)
    timestamps = pd.to_datetime(raw["@timestamp"],utc=True,format="ISO8601")
    if not (raw["@timestamp"].str[-6:].eq("+01:00").all() and timestamps.notna().all()):
        raise ValueError("Unexpected clock offset in MSDS aggregate log")
    eligible = raw.Logger.str.fullmatch(r"openstack\.[a-z0-9_-]+",na=False)
    ticks = timestamps.dt.floor("s")
    matrix = pd.crosstab(ticks[eligible],raw.loc[eligible,"Logger"])
    index = pd.date_range(ticks.min(),ticks.max(),freq="s")
    matrix = matrix.reindex(index,fill_value=0).astype("int32")
    matrix.columns = matrix.columns.tolist()
    matrix.index.name = "utc_time"
    matrix.to_csv(folder/"observations.csv",date_format="%Y-%m-%dT%H:%M:%S%z")
    return {"input_zip_member":LOG,"input_sha256":sha256(z.read(LOG)).hexdigest(),
            "input_log_rows":len(raw),"unassigned_log_rows":int((~eligible).sum()),
            "channels":matrix.columns.tolist(),"messages_per_service":matrix.sum().to_dict(),
            "start_utc":index[0].isoformat(),"end_bin_start_utc":index[-1].isoformat(),
            "one_second_bins":len(index),"all_service_zero_bins":int((matrix.sum(axis=1)==0).sum())}


def hook_ledger(z,channels):
    rows=[]
    for report in REPORTS:
        name=PREFIX+"reports/output_"+report+".json"
        entry=json.loads(z.read(name))
        workloads=entry["tasks"][0]["subtasks"][0]["workloads"]
        if len(workloads)!=1 or len(workloads[0]["hooks"])!=1:
            raise ValueError("Unexpected Rally report workload/hook structure")
        hook=workloads[0]["hooks"][0]
        for j,r in enumerate(hook["results"]):
            text=r["output"]["complete"][0]["data"]
            stdout=next(s for s in text if s.startswith("StdOut:"))
            containers=sorted(set(re.findall(r"^(?:nova|glance|neutron)_[a-z0-9_]+$",stdout,re.M)))
            services=sorted({"openstack."+c.split("_")[0] for c in containers})
            if r["status"]!="success" or "RetCode: 0" not in text or len(services)!=1:
                raise ValueError(f"Unverified hook target or failed hook in {name}[{j}]")
            if not set(services).issubset(channels):
                raise ValueError(f"Hook target outside observed services: {services}")
            start,stop=r["started_at"],r["finished_at"]
            if not stop>start:
                raise ValueError("Invalid Rally hook times")
            rows.append({"id":f"{report}-iteration-{r['triggered_by']['value']}","source":name,
                         "evidence_ref":f"{name}#tasks[0].subtasks[0].workloads[0].hooks[0].results[{j}]",
                         "status":r["status"],"retcode":0,"containers":containers,"services":services,
                         "host_ids_in_result":sorted(set(re.findall(r'\bwally\d+\b',str(r)))),
                         "start_epoch":start,"stop_epoch":stop,"start_utc":iso(start),"stop_utc":iso(stop)})
    rows.sort(key=lambda r:(r["start_epoch"],r["id"]))
    if len(rows)!=31:
        raise ValueError("Release hook count mismatch")
    return rows


def merged_events(hooks,lo,hi):
    groups=[]
    for h in sorted(hooks,key=lambda row:(row["start_epoch"],row["id"])):
        if not (lo<=h["start_epoch"]<h["stop_epoch"]<=hi):
            raise ValueError("Hook outside independent log collection")
        if groups and h["start_epoch"] < max(k["stop_epoch"] for k in groups[-1]):
            groups[-1].append(h)
        else:
            groups.append([h])
    events=[]
    for hs in groups:
        a=ceil(min(h["start_epoch"] for h in hs))
        b=floor(max(h["stop_epoch"] for h in hs))
        if not a<b:
            raise ValueError("No full observation bin inside hook interval")
        events.append({"id":"+".join(h["id"] for h in hs),"start_time":iso(a),"end_time":iso(b),
                       "end_inclusive":False,"status":"eligible","targets":sorted({s for h in hs for s in h["services"]}),
                       "evidence_ref":" ; ".join(h["evidence_ref"] for h in hs),"hook_ids":[h["id"] for h in hs]})
    return events


def match_clock(z):
    # Audit only: shared resource ID, request method, service, and hostname identify a
    # single independent API-completion log. Neither raw message nor request ID is a feature.
    logs=pd.read_csv(z.open(LOG),usecols=["@timestamp","Hostname","Logger","Payload","message"],low_memory=False)
    leakage = r"restart_(?:nova|glance|neutron)_container|Restarting (?:nova|glance|neutron)-container|System call|RetCode: 0"
    echo=int((logs.Payload.str.contains(leakage,case=False,na=False)|logs.message.str.contains(leakage,case=False,na=False)).sum())
    if echo:
        raise ValueError("Aggregate service log contains possible injection-harness echoes")
    scoped=logs[logs.Logger.eq("openstack.glance")].dropna(subset=["Payload"]).copy()
    scoped["key"]=scoped.Payload.str.extract(r'"DELETE (/v2/images/[0-9a-f-]{36})\s',expand=False)
    scoped=scoped.dropna(subset=["key"]).set_index(["Hostname","key"])
    w=json.loads(z.read(PREFIX+"reports/output_image.json"))["tasks"][0]["subtasks"][0]["workloads"][0]
    iterations=sorted(w["data"],key=lambda r:r["timestamp"])
    anchors=[]
    for at in np.linspace(0,len(iterations)-50,12,dtype=int):
        for r in iterations[at:at+50]:
            if r["error"]:continue
            ident=r["output"]["complete"][0]["data"]["trace_id"]
            member=PREFIX+f"traces/image_create_delete/{ident}.json"
            stack=[json.loads(z.read(member))]
            hit=None
            while stack and not hit:
                node=stack.pop(); stack.extend(node.get("children",[])); info=node["info"]
                begin=info.get("meta.raw_payload.wsgi-start",{});end=info.get("meta.raw_payload.wsgi-stop",{})
                req=begin.get("info",{}).get("request",{})
                if info.get("project")!="glance" or req.get("method")!="DELETE" or not end:continue
                key=(info["host"],req["path"])
                if key not in scoped.index:continue
                matched=scoped.loc[[key]]
                if len(matched)!=1:continue
                row=matched.iloc[0]
                trace=float(pd.Timestamp(end["timestamp"],tz="UTC").timestamp())
                log=float(pd.Timestamp(row["@timestamp"]).timestamp())
                if abs(log-trace)>2 or not r["timestamp"]-2<=trace<=r["timestamp"]+r["duration"]+2:continue
                hit={"report_iteration_start_utc":iso(r["timestamp"]),"report_iteration_stop_utc":iso(r["timestamp"]+r["duration"]),
                     "trace_member":member,"trace_completion_utc":iso(trace),
                     "resource_path_sha256":sha256(req["path"].encode()).hexdigest(),"host":info["host"],
                     "native_log_time":row["@timestamp"],"log_minus_trace_seconds":round(log-trace,6)}
            if hit:
                anchors.append(hit);break
    if len(anchors)!=12:
        raise ValueError(f"Insufficient cross-source clock validation: {len(anchors)} anchors")
    return {"anchors":anchors,"injector_echo_signature_matches_in_log":echo,
            "max_abs_completion_disagreement_seconds":max(abs(a["log_minus_trace_seconds"]) for a in anchors)}


def prepare_service(data):
    """Build direct service targets and independent log-count measurements into separate release artifacts."""
    data = Path(data)
    folder = data/"msds-direct-services"
    folder.mkdir(parents=True,exist_ok=True)
    url = "https://zenodo.org/api/records/3549604/files/concurrent%20data.zip/content"
    archive = download(url,data/"archives"/"msds-concurrent.zip","be2ba2f17f774e7f3f045dcf5f406ff0","md5")
    with zipfile.ZipFile(archive) as z:
        info=observed(z,folder)
        hooks=hook_ledger(z,info["channels"])
        lo=pd.Timestamp(info["start_utc"]).timestamp()
        hi=pd.Timestamp(info["end_bin_start_utc"]).timestamp()+1
        events=merged_events(hooks,lo,hi)
        clocks=match_clock(z)
    save_json(folder/"rally_hooks.json",hooks);save_json(folder/"events.json",events)
    save_json(folder/"clock_audit.json",clocks);save_json(folder/"feature_audit.json",info)
    spec={"dataset":"MSDS","release":"Zenodo 3549604 v1.1 concurrent, Rally hooks and independent service log counts",
          "id":"msds-direct-service-logs","group":"concurrent-2019-11-25","truth_unit":"direct-injected-service",
          "source_url":REF,"features":{s:s for s in info["channels"]},"candidate_ids":info["channels"],
          "recording":{"path":"observations.csv","time_columns":["utc_time"],"time_format":"%Y-%m-%dT%H:%M:%S%z","timezone":"UTC"},
          "events_json":"events.json"}
    save_json(folder/"spec.json",spec)
    industrial_import(folder/"spec.json",data)
    ledger=data/"rca/msds-direct-service-logs.json";item=json.loads(ledger.read_text(encoding="utf8"))[0]
    if len(item["events"])!=29 or len(hooks)!=31:
        raise ValueError("Unexpected number of registered MSDS service events")
    for e in item["events"]:
        if e["stop"]-e["start"]!=int((pd.Timestamp(e["end_time"])-pd.Timestamp(e["start_time"])).total_seconds()):
            raise ValueError("Row alignment does not match interior one-second bins")
    print("31 successful hooks ->",len(events),"direct-service events; clock max seconds",clocks["max_abs_completion_disagreement_seconds"])
    print(ledger)
