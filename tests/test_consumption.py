import json
from types import SimpleNamespace

import numpy as np
import pytest

from shifted_attention.consumption import QUALITY, counts_by_cell
from shifted_attention.data import DTYPE


class FixtureCorpus(SimpleNamespace):
    def __len__(self):
        return len(self.table)


def test_repeat_visits_witnessed_events_and_known_slots(tmp_path):
    receipts = tmp_path/"receipts"
    receipts.mkdir()
    names = ["a"*32, "b"*32]
    table = np.zeros(2,dtype=DTYPE)
    table["id"] = names
    path = "shards/n004-0000000.tar"
    corpus = FixtureCorpus(split=tmp_path,table=table,meta={"shards":[path]})
    common = dict(N=4,K=2,mode="mixed",raw=[0,1],known=[0],
                  quality={name:False for name in QUALITY},focal=["process","L01",0])
    rows = [{**common,"id":names[0],"events":[["process","L01",False,0],["process","L01",True,1],
                                               ["observation","L01",False,0]],"focal_witness":False},
            {**common,"id":names[1],"known":[1],"events":[["process","L01",True,0],
                                                           ["observation","L01",True,1]],"focal_witness":True}]
    receipt = receipts/"n004-0000000.json"
    receipt.write_text(json.dumps(dict(rows=rows,public=path)))
    keys = [[4,2,"mixed",layer,"L01",1] for layer in ("process","observation")]
    released = dict(cells=[[4,2,"mixed",2]],event_counts=keys,focal_counts=keys[:1])
    report = counts_by_cell(corpus, [1,0,0], released)
    by_layer = {r["layer"]:r for r in report["templates"]}
    assert report["root_exposure"][4] == dict(raw=[3,3,0,0],known=[2,1,0,0])
    assert {k:by_layer["process"][k] for k in ("events","events_witnessed","windows","windows_witnessed","designated","designated_witnessed")} == dict(
        events=5,events_witnessed=3,windows=3,windows_witnessed=3,designated=3,designated_witnessed=1)
    assert by_layer["observation"]["windows_witnessed"] == 1
    assert report["cells"][0]["quality"]["no_visible_root"] == dict(count=0,denominator=3,fraction=0.)
    rows[0].pop("quality")
    rows[1].pop("quality")
    for row in rows:
        row["events"] = [e[:3] for e in row["events"]]
    receipt.write_text(json.dumps(dict(rows=rows,public=path)))
    legacy = counts_by_cell(corpus,[0],{**released,"events_with_witnessed_root":[]})
    assert legacy["template_witness_definition"] == "target_root_witness_in_window"
    assert legacy["cells"][0]["quality"]["no_visible_root"] == dict(count=None,denominator=0,fraction=None)
    with pytest.raises(ValueError,match="missing declared visibility"):
        counts_by_cell(corpus,[0],{**released,"quality_flags":[]})
    rows.reverse()
    receipt.write_text(json.dumps(dict(rows=rows,public=path)))
    with pytest.raises(ValueError,match="receipt ID"):
        counts_by_cell(corpus,[0],released)
