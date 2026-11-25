"""Caps on per-seed sampling and rejection of partial or corrupted corpus releases."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from shifted_attention.data import apportion,build_index,consumption_order,DTYPE,PROFILE,validation_halves


def test_budget_is_exact_stratified_and_without_replacement():
    sizes=[(1,1,1,0,957), (1,1,1,1,43), (4,2,3,0,753),
           (4,2,3,41,97), (4,2,3,2,50), (128,0,0,0,800)]
    rows=np.zeros(sum(v[-1] for v in sizes),dtype=DTYPE)
    pos=0
    for n,k,mode,f,amount in sizes:
        part=rows[pos:pos+amount]
        for key,value in zip(("n","k","mode","focal"),(n,k,mode,f)):
            part[key]=value
        pos+=amount
    order=consumption_order(rows,43,731)
    assert np.array_equal(np.sort(order),np.arange(len(rows)))
    assert np.array_equal(order,consumption_order(rows,43,731))
    assert not np.array_equal(order,consumption_order(rows,44,731))
    used=rows[order[:731]]
    quotas=apportion([1000,900,800],731)
    for n,seats in zip([1,4,128],quotas):
        assert (used["n"]==n).sum()==seats
        _, original=np.unique(rows["focal"][rows["n"]==n],return_counts=True)
        _, actual=np.unique(used["focal"][used["n"]==n],return_counts=True)
        np.testing.assert_array_equal(actual,apportion(original,seats))


def test_validation_groups_all_descendants_by_complete_ancestor_key():
    t=np.zeros(750*2*4,dtype=DTYPE)
    t["n"]=8
    t["family"]=np.repeat(np.arange(4),1500)
    t["ancestor"]=np.tile(np.repeat(np.arange(750),2),4)
    select,cal=validation_halves(t)
    assert len(select)==len(cal)==len(t)//2
    for f in range(4):
        left,right=(set(a["ancestor"][a["family"]==f]) for a in (t[select],t[cal]))
        assert not left&right and len(left)==len(right)==375


def test_reader_checks_published_hash_and_rebuilds_missing_index(tmp_path):
    row={"id":"1"*32,"N":4,"K":1,"mode":"process","family":"linear","ancestor":7,"focal":None,
         "shard":"shards/a.tar","input_offset":512,"input_size":10,"target_offset":1536,"target_size":20}
    raw=(json.dumps(row)+"\n").encode()
    (tmp_path/"manifest.jsonl").write_bytes(raw)
    (tmp_path/"audit.json").write_bytes(b"{}")
    release=dict(profile=PROFILE,windows=1,smoke=True,manifest_sha256=hashlib.sha256(raw).hexdigest(),
                 audit_sha256=hashlib.sha256(b"{}").hexdigest())
    (tmp_path/"_SUCCESS").write_text(json.dumps(release))
    build_index(tmp_path)
    meta=json.loads((tmp_path/"reader-index.json").read_text())
    indexed=tmp_path/meta["data"]
    indexed.unlink()
    build_index(tmp_path)
    assert indexed.stat().st_size==DTYPE.itemsize
    (tmp_path/"manifest.jsonl").write_bytes(raw.replace(b"linear",b"additive"))
    with pytest.raises(ValueError,match="manifest differs"):
        build_index(tmp_path)
