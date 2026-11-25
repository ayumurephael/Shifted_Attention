import csv
import gzip
import json
import sqlite3

import numpy as np
import pytest
import torch
from torch import nn

from shifted_attention.data import DTYPE
from shifted_attention.validation import (SCHEMA, TiedScores, _combine, _score_summary,
                                          average_precision, validate)


class ToyCorpus:
    def __init__(self):
        self.table = np.zeros(6,dtype=DTYPE)
        self.table["n"] = [1,4,4,1,4,4]
        self.table["mode"] = [0,0,1,2,3,2]
        self.table["family"] = [0,1,2,3,1,2]
        self.table["k"] = [0,0,2,1,2,1]
        self.table["ancestor"] = np.arange(6)
        self.table["id"] = [f"event-{i}".encode() for i in range(6)]

    def __len__(self):
        return 6

    def __getitem__(self,i):
        n = int(self.table["n"][i])
        p = torch.tensor([[.1,.2],[.2,.4],[.8,.5,.9],[.4,.6],[.7,.1],[.2,.3]][i])
        t = len(p)
        x = torch.logit(p)[:,None].expand(t,n).clone()
        obs = torch.ones_like(x,dtype=torch.bool)
        ry = torch.zeros(n,dtype=torch.bool)
        rk = torch.ones(n,dtype=torch.bool)
        yk = torch.ones(t,dtype=torch.bool)
        yy = torch.zeros(t,dtype=torch.bool)
        if i==2:
            obs[:,-1]=False
            obs[-1]=False
            ry[[0,3]]=True
            rk[2]=False
            yy[[0,2]]=True
        if i==3:
            ry[:]=True
            yy[:]=True
        if i==4:
            rk[:]=False
            yk[:]=False
        return dict(index=int(i),values=x,observed=obs,time=torch.arange(t).double(),
                    channel_valid=torch.ones(n,dtype=torch.bool),R=ry,R_known=rk,E=torch.ones_like(ry),
                    Y=yy,Y_known=yk)


class ToyModel(nn.Module):
    def forward(self,values,observed,time,channel_valid,row_valid):
        return dict(root_logits=torch.zeros_like(values[:,0]),root_valid=observed.any(1),
                    anomaly_logits=values[:,:,0],anomaly_valid=observed.any(-1))


def test_all_three_protocols(tmp_path):
    dataset = ToyCorpus()
    model = ToyModel()
    for purpose in ("selection","calibration","test"):
        report = validate(model,dataset,np.arange(len(dataset)),tmp_path,torch.device("cpu"),2,0,purpose)
        assert json.loads((tmp_path/f"{purpose}.json").read_text()) == report
        assert model.training
        assert report["root_map"] == pytest.approx(.25+1/3)
        score = np.array([.1,.2,.2,.4,.8,.5,0,.4,.6,.2,.3],dtype=np.float32)
        y = np.array([0,0,0,0,1,0,1,1,1,0,0])
        assert report["point_ap"] == pytest.approx(average_precision(score,y))
        assert report["point_pooled"]["brier"] == pytest.approx(np.mean((score-y)**2))
        assert sum(b["count"] for b in report["point_pooled"]["bins"])==len(y)
        d = report["denominators"]
        assert d["all_unknown_root_windows"]==1 and d["all_positive_root_windows"]==1
        assert d["healthy_windows"]==2 and d["all_negative_root_windows"]==3
        assert d["known_rows_without_evidence"]==1 and d["positive_roots_without_evidence"]==1
        assert report["window_metrics"]["by_K"]["2"]["rankable_windows"]==1
        assert report["point_metrics"]["by_mode"]["mixed"]["known"]==0
        assert report["selection_score"] == pytest.approx((report["root_map"]+report["point_ap"])/2)
        if purpose=="calibration":
            assert report["threshold"]==pytest.approx(.39)
            assert report["threshold_windows"]==2
        else:
            assert report["threshold"] is None
        if purpose=="test":
            with gzip.open(tmp_path/report["event_records"],"rt") as f:
                rows=list(csv.DictReader(f))
            assert [r["window_id"] for r in rows]==[f"event-{i}" for i in range(6)]
            assert rows[2]["mode"]=="process" and float(rows[2]["root_ap"])==report["root_map"]


def test_merged_rank_ties_and_flushed_fragments(tmp_path):
    paths=[]
    chunks=[([1.,.8,.4],[1,0,1]),([.8,.4,0.],[1,0,1])]
    for i,(s,y) in enumerate(chunks):
        path=tmp_path/f"rank-{i}.db"
        paths.append(path)
        with sqlite3.connect(path) as db:
            db.executescript(SCHEMA)
            h=TiedScores(db,1)
            for score,label in zip(s,y):
                h.add(np.array([score],dtype=np.float32),np.array([label]),4,1)
                h.flush()
    with sqlite3.connect(paths[0]) as db:
        _combine(db,paths[1])
        got=_score_summary(db,1,calibration=True)
    assert got["known"]==6 and got["positive"]==4
    assert got["ap"]==pytest.approx((1+2/3+3/5+4/6)/4)
    assert got["bins"][-1]["count"]==1
