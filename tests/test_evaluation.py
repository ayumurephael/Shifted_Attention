import dataclasses
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from shifted_attention.config import ModelConfig
from shifted_attention.model import ShiftedAttentionModel
from shifted_attention.evaluation.adapters import industrial_import,recording,segments
from shifted_attention.evaluation.assets import git_blob
from shifted_attention.evaluation.baselines import Severity
from shifted_attention.evaluation.msds_services import LOG, merged_events, observed
from shifted_attention.evaluation.inference import Predictor,normalizer,starts
from shifted_attention.evaluation.rca import aggregate,match_events,ranking,score_record
from shifted_attention.evaluation.tsad import PAPER,file_info,summarize


def test_protocol_inventory_and_indices():
    assert [sum(x.values()) for x in PAPER.values()] == [700,68]
    assert file_info("001_UCR_id_5_Domain_tr_100_1st_181.csv") == ("UCR",100)
    assert starts(15,8,6)==[0,6,7]
    assert starts(2,8,6)==[0]
    assert segments([1,0,1,1,0,1])==[(0,1),(2,4),(5,6)]
    median,scale = normalizer(np.array([[1.,np.nan],[1.,np.nan]]))
    np.testing.assert_array_equal(median,[1,0])
    np.testing.assert_array_equal(scale,[1,1])
    assert git_blob(b"")== "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_undefined_metrics_keep_series_and_source_denominators():
    rows = [dict(split="U",dataset="a",metrics={"m":0.3}),
            dict(split="U",dataset="a",metrics={"m":None}),
            dict(split="U",dataset="b",metrics={"m":0.7})]
    value = summarize(rows)
    assert value["dataset"]["U-a"]["files"]==2
    assert value["dataset"]["U-a"]["finite_denominator"]["m"]==1
    assert value["dataset_equal"]["metrics"]["m"]==pytest.approx(.5)


def test_severity_uses_identical_reference_and_retains_unobserved_candidates():
    x=np.array([[0.,np.nan,0.],[1.,np.nan,3.],[np.nan,np.nan,np.nan]])
    ref=np.zeros((2,3))
    model=Severity()
    score,mask,n=model.root(x,np.arange(3),0,3,ref)
    np.testing.assert_allclose(score,[.5,0,.75])
    assert mask.tolist()==[True,False,True] and n==1
    scores,hits=model.anomaly(x,np.arange(3),ref=ref,width=2,stride=1)
    np.testing.assert_allclose(scores,[0,.75,0])
    assert hits.tolist()==[1,2,0]


def test_independent_service_logs_and_overlapping_hook_interiors(tmp_path):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w") as archive:
        archive.writestr(LOG,"@timestamp,Logger\n2019-11-25T16:00:00+01:00,openstack.a\n"
                        "2019-11-25T16:00:00+01:00,openstack.b\n2019-11-25T16:00:02+01:00,\n")
    with zipfile.ZipFile(stream) as archive:
        report=observed(archive,tmp_path)
    assert report["channels"]==["openstack.a","openstack.b"] and report["one_second_bins"]==3
    assert report["start_utc"]=="2019-11-25T15:00:00+00:00" and report["unassigned_log_rows"]==1
    hooks=[dict(id="b",start_epoch=3.2,stop_epoch=6.8,services=["b"],evidence_ref="y"),
           dict(id="a",start_epoch=1.1,stop_epoch=4.3,services=["a"],evidence_ref="x")]
    incidents=merged_events(hooks,0,10)
    assert len(incidents)==1 and incidents[0]["targets"]==["a","b"]
    assert incidents[0]["end_inclusive"] is False
    assert incidents[0]["start_time"].endswith("00:00:02+00:00")
    assert incidents[0]["end_time"].endswith("00:00:06+00:00")


def test_ap_ties_multiroot_and_cardinality_first():
    metrics = ranking(["a","b","c"],[0.5,0.5,0.5],["a","b"],"release","run")
    assert metrics["AP"]==pytest.approx(2/3)
    assert metrics["Recall@3"]==1
    true = [{"start":0,"stop":10,"id":"early"},{"start":10,"stop":20,"id":"late"}]
    assert match_events(true,[(0,2),(3,15)])=={0:0,1:1}
    assert match_events(true,[(22,30)])=={}
    assert match_events([{"start":0,"stop":10,"id":"a"}],[(0,2),(8,10)])=={0:0}


def test_release_import_sorts_duplicate_timestamps_and_checks_signed_labels(tmp_path):
    (tmp_path/"measure.csv").write_text(
        "Date,Time,X,Y,Flag\n2019-12-19,00:00:02,2,3,1\n2019-12-19,00:00:01,3,0,-1\n"
        "2019-12-19,00:00:01,5,2,1\n",encoding="utf8")
    events = [dict(id="injection",start_time="2019-12-19T00:00:01",end_time="2019-12-19T00:00:01",
                   status="eligible",targets=["x"],evidence_ref="synthetic fixture row 1")]
    (tmp_path/"targets.json").write_text(json.dumps(events))
    spec = dict(dataset="WADI",id="release-test",group="one",release="fixture",truth_unit="direct",
                source_url="fixture",features={"x":"X","y":"Y"},candidate_ids=["x","y"],
                recording=dict(path="measure.csv",time_columns=["Date","Time"],time_format="%Y-%m-%d %H:%M:%S",
                               timezone="Asia/Singapore",label_column="Flag",label_mapping={"1":0,"-1":1}),
                events_json="targets.json")
    file=tmp_path/"spec.json"
    file.write_text(json.dumps(spec))
    industrial_import(file,tmp_path)
    rec=json.loads((tmp_path/"rca/release-test.json").read_text())[0]
    assert rec["events"][0]["stop"]==1 and rec["label_disagreement_rows"]==0
    with np.load(tmp_path/rec["file"]) as series:
        np.testing.assert_array_equal(series["x"], [[4,1],[2,3]])


def test_two_heads_checkpoint_and_event_scoring(tmp_path):
    cfg=ModelConfig(stream_width=12,blocks=1,patch_chunk_size=2)
    model=ShiftedAttentionModel(cfg)
    ckpt=tmp_path/"checkpoint.pt"
    torch.save({"model_config":dataclasses.asdict(cfg),"model":model.state_dict()},ckpt)
    predictor=Predictor(ckpt,"cpu")
    x=np.arange(45,dtype="float32").reshape(15,3)
    x[4]=np.nan
    x[:,2]=np.nan
    t=np.arange(15,dtype=float)
    scores,hits=predictor.anomaly(x,t,width=8,stride=6)
    assert len(scores)==15 and scores[4]==0 and hits[4]==0 and np.all(hits[np.arange(15)!=4]>0)
    columns=["a","b","c"]
    rec=recording(tmp_path,"raw",x,columns,t)
    rec.update(dataset="fixture",release="r1",id="raw",group="run1",truth_unit="measured",candidate_ids=columns,
               events=[dict(id="one",start=1,stop=5,targets=["a"],status="eligible")])
    rows,alarm=score_record(predictor,tmp_path,rec,1.)
    assert rows[0]["scores"][-1]==0 and rows[0]["evidence"][-1] is False
    assert rows[1]["metrics"]["MRR"]==0 and alarm["matched"]==0
    assert aggregate(rows)["ranking"]["fixture|measured|given-event"]["events"]==1
    _,zero_alarm=score_record(predictor,tmp_path,rec,0.)
    assert zero_alarm["intervals"]==[(0,4),(5,15)]
