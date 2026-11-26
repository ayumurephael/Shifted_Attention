import hashlib
import json
import pytest

from shifted_attention.data import Corpus, PROFILE, PRODUCTION_WIDTHS, ROUND1_EDITION, build_index, production_edition
from shifted_attention.train import check_checkpoint_contract


def publish_stub(root, split, per_width, edition=None, calibration="c"*64):
    folder=root/split
    folder.mkdir(exist_ok=True)
    windows=per_width*len(PRODUCTION_WIDTHS)
    marker=dict(profile=PROFILE,windows=windows,smoke=False,manifest_sha256="a"*64)
    if edition:
        marker["training_edition"]=edition
    audit=dict(marker,split=split,calibration_sha256=calibration,
               cells=[[n,0,"healthy",per_width] for n in PRODUCTION_WIDTHS])
    write_pair(folder,marker,audit)
    return marker,audit


def write_pair(folder,marker,audit):
    raw=json.dumps(audit).encode()
    (folder/"audit.json").write_bytes(raw)
    marker["audit_sha256"]=hashlib.sha256(raw).hexdigest()
    (folder/"_SUCCESS").write_text(json.dumps(marker))


def test_full_and_round1_require_matching_release_and_heldout(tmp_path):
    for split in ("validation","test"):
        publish_stub(tmp_path,split,48_000)
    publish_stub(tmp_path,"train",960_000)
    assert production_edition(tmp_path) is None
    marker,audit=publish_stub(tmp_path,"train",240_000,ROUND1_EDITION)
    assert production_edition(tmp_path)==ROUND1_EDITION

    del audit["training_edition"]
    write_pair(tmp_path/"train",marker,audit)
    with pytest.raises(ValueError,match="inconsistent training edition"):
        production_edition(tmp_path)
    publish_stub(tmp_path,"train",240_000)
    with pytest.raises(ValueError,match="unexpected corpus counts"):
        production_edition(tmp_path)
    marker,audit=publish_stub(tmp_path,"train",240_000,ROUND1_EDITION)
    audit["cells"][0][-1]-=1
    audit["cells"][1][-1]+=1
    write_pair(tmp_path/"train",marker,audit)
    with pytest.raises(ValueError,match="unexpected corpus counts"):
        production_edition(tmp_path)
    publish_stub(tmp_path,"train",240_000,ROUND1_EDITION)
    publish_stub(tmp_path,"test",48_000,ROUND1_EDITION)
    with pytest.raises(ValueError,match="only to a formal train"):
        production_edition(tmp_path)
    publish_stub(tmp_path,"test",48_000,calibration="d"*64)
    with pytest.raises(ValueError,match="calibration hashes differ"):
        production_edition(tmp_path)


def test_cross_edition_checkpoint_and_index_rejected(tmp_path):
    # Even an identical hash vector cannot silently cross a declared edition.
    old={"dataset":["a"*64,"b"*64,"c"*64],"steps":60000,"global_batch":24}
    new=dict(old,training_edition=ROUND1_EDITION)
    check_checkpoint_contract({"contract":old},old)
    check_checkpoint_contract({"contract":new},new)
    for checkpoint,contract in [(old,new),(new,old),(dict(new,dataset=["e"*64,*new["dataset"][1:]]),new)]:
        with pytest.raises(ValueError,match="resume contract differs"):
            check_checkpoint_contract({"contract":checkpoint},contract)

    folder=tmp_path/"train"
    marker,audit=publish_stub(tmp_path,"train",240_000,ROUND1_EDITION)
    meta={"manifest_sha256":marker["manifest_sha256"],"count":marker["windows"],
          "data":"reader-"+marker["manifest_sha256"]+".bin"}
    (folder/"reader-index.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="index does not belong"):
        Corpus(folder)
    marker["training_edition"]="unknown-edition"
    audit["training_edition"]="unknown-edition"
    write_pair(folder,marker,audit)
    with pytest.raises(ValueError,match="unrecognized"):
        build_index(folder)


def test_new_index_binds_edition_and_verifies_real_row_widths(tmp_path,monkeypatch):
    from shifted_attention import data

    monkeypatch.setattr(data,"ROUND1_PER_WIDTH",2)
    folder=tmp_path/"train"
    marker,audit=publish_stub(tmp_path,"train",2,ROUND1_EDITION)
    rows=[]
    for index,n in enumerate(PRODUCTION_WIDTHS):
        for j in range(2):
            rows.append(dict(id=f"{2*index+j:032x}",N=n,K=0,mode="healthy",family="linear",ancestor=j,
                             focal=None,shard="shards/tiny.tar",input_offset=512,input_size=1,
                             target_offset=1536,target_size=1))
    def manifest():
        payload=b"".join(json.dumps(r).encode()+b"\n" for r in rows)
        (folder/"manifest.jsonl").write_bytes(payload)
        audit["manifest_sha256"]=marker["manifest_sha256"]=hashlib.sha256(payload).hexdigest()
        write_pair(folder,marker,audit)
    manifest()
    build_index(folder)
    reader=Corpus(folder)
    assert reader.meta["training_edition"]==ROUND1_EDITION
    assert list(reader.table["n"])==[r["N"] for r in rows]
    del reader
    meta_path=folder/"reader-index.json"
    stale=json.loads(meta_path.read_text())
    del stale["training_edition"]
    meta_path.write_text(json.dumps(stale))
    build_index(folder)
    assert json.loads(meta_path.read_text())["training_edition"]==ROUND1_EDITION
    rows[0]["N"]=128
    manifest()
    with pytest.raises(ValueError,match="manifest channel-width counts"):
        build_index(folder)
