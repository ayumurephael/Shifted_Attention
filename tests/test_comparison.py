import csv
import json

import numpy as np
import pytest
import torch

from shifted_attention.comparison import PointAP, compare, holm, real_rca, run, synthetic, tsb
from shifted_attention.validation import average_precision, validate
from test_validation import ToyCorpus, ToyModel


def test_exact_clustered_point_ap_matches_expanded_raw_points(tmp_path):
    ds=ToyCorpus()
    out=tmp_path/"synthetic-test.json"
    scored=validate(ToyModel(),ds,np.arange(len(ds)),out,torch.device("cpu"),2,0,"test")
    p=synthetic(out,{"metric":"point_ap"})
    groups,indices=np.unique(p.groups,return_inverse=True)
    engine=PointAP(p,groups,"cpu")
    np.testing.assert_allclose(engine.values(np.ones((1,len(groups))))[0],scored["point_ap"])
    w=np.array([2,0,3,1,2,4],dtype=float)
    pred,truth=[],[]
    for j,idx in enumerate(indices):
        record=ds[j]
        obs=record["observed"].any(1)
        exact=torch.where(obs,record["values"][:,0].sigmoid(),0.0).numpy()
        mask=record["Y_known"].numpy()
        pred.extend([exact[mask]]*int(w[idx]))
        truth.extend([record["Y"].numpy()[mask]]*int(w[idx]))
    assert engine.values(w[None])[0]==pytest.approx(average_precision(np.concatenate(pred),np.concatenate(truth)))
    spec={"seeds":[17],"systems":{"full":{"17":{"synthetic":out.name}},
                                  "sync":{"17":{"synthetic":out.name}}},
          "contrasts":[{"name":"shift","left":"full","right":"sync","analyses":[
               {"name":"point","kind":"synthetic","metric":"point_ap","holm_family":"A"},
               {"name":"root","kind":"synthetic","metric":"root_map","holm_family":"A"}]}]}
    manifest=tmp_path/"comparisons.json"
    manifest.write_text(json.dumps(spec))
    got=run(manifest,tmp_path/"report.json")
    assert got["comparisons"]["shift/point"]["estimate"]==0
    assert got["comparisons"]["shift/point"]["interval"] is None
    assert got["comparisons"]["shift/point"]["requested"]==10000
    # There is only one rankable window: many resamples have no rankable units.
    assert got["comparisons"]["shift/root"]["finite"]<9900
    assert got["comparisons"]["shift/root"]["interval"] is None


def test_joint_seeds_and_physical_record_groups(tmp_path):
    methods=[tmp_path/f"arm-{i}" for i in range(4)]
    files=["a.csv","b.csv","c.csv","d.csv","e.csv"]
    for arm,d in enumerate(methods):
        (d/"M").mkdir(parents=True)
        for j,name in enumerate(files):
            if arm in (1,3) and j==4:
                continue
            score=(.1 if arm in (0,1) else .2)+(arm%2)*[.1,-.1,.2,-.1,.0][j]
            row=dict(identity=dict(protocol="tsb",source_sha256="source-"+name,metric_commit="rev"),
                     split="M",file=name,dataset="S",start_row_retained=0,n=10,positive_rows=1,
                     normalization="normal-reference",period=1,metrics={"VUS-PR":score})
            (d/"M"/(name+".json")).write_text(json.dumps(row))
    mapping={("M","a.csv"):"derived",("M","b.csv"):"derived"}
    spec={"kind":"tsb","split":"M","metric":"VUS-PR","margin":-.01}
    v=[tsb(d,spec,mapping) for d in methods]
    report=compare([(v[0],v[1]),(v[2],v[3])],[17,29],spec,replicates=10000)
    assert report["groups"]==4
    assert report["seed_results"]["17"]["coverage"]["paired_finite"]==4
    assert report["seed_results"]["17"]["coverage"]["left_finite"]==5
    assert report["estimate"]==pytest.approx(-.025)
    assert report["finite"]<10000 and report["finite"]>=9900
    assert report["interval"][0]<report["estimate"]<report["interval"][1]


def test_experiment_equal_unequal_event_counts(tmp_path):
    def output(file,positive):
        rows=[]
        for i,n in enumerate([1,3,1]):
            for j in range(n):
                r=dict(dataset="light",truth_unit="device-owner-proxy",mode="given-event",
                       release="v1",recording=str(i),event=str(j),group=str(i),status="eligible",
                       start=j,stop=j+1,targets=["a"],candidates=["a","b"],
                       metrics={"AP":float(i==1 and positive)})
                rows.append(r)
        file.write_text(json.dumps(rows))
    spec=dict(kind="rca",dataset="light",truth_unit="device-owner-proxy",mode="given-event",metric="AP")
    a,b=tmp_path/"left.json",tmp_path/"right.json"
    output(a,True)
    output(b,False)
    result=compare([(real_rca(a,spec),real_rca(b,spec))],[17],spec)
    assert result["estimate"]==pytest.approx(1/3)
    assert len(result["per_experiment"])==3
    assert holm({"a":.02,"b":.001,"c":None})=={"a":.04,"b":.003,"c":None}
