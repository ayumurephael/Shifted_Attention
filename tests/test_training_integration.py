"""Real-generator integration tests. Set SSTER_INTEGRATION_DATA to a three-split smoke release."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest
import torch

from shifted_attention.data import Corpus, consumption_order
from shifted_attention.train import publish_checkpoint


def test_checkpoint_without_hardlink(tmp_path,monkeypatch):
    source=tmp_path/"step.pt"
    destination=tmp_path/"latest.pt"
    source.write_bytes(b"checkpoint payload")
    destination.write_bytes(b"previous state")
    def denied(*_args):
        raise PermissionError("hard links unavailable")
    monkeypatch.setattr(os,"link",denied)
    publish_checkpoint(source,destination)
    assert destination.read_bytes()==source.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir())==["latest.pt","step.pt"]


@pytest.fixture(scope="module")
def corpus():
    path=os.getenv("SSTER_INTEGRATION_DATA")
    if not path:
        pytest.skip("SSTER_INTEGRATION_DATA is unset")
    path=Path(path).resolve()
    for s in ("train","validation","test"):
        assert (path/s/"_SUCCESS").is_file()
    return path


def command(corpus,output):
    return [sys.executable,"-m","shifted_attention.train","--dataset",str(corpus),
            "--output",str(output),"--fixture","--device","cpu","--workers","0",
            "--steps","2","--interval","1","--warmup","1","--global-batch","4",
            "--microbatch","1","--eval-batch","2"]


def launch_env():
    return {**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
            "PYTHONUTF8":"1","PYTHONIOENCODING":"utf-8","USE_LIBUV":"0"}


@pytest.fixture(scope="module")
def single(corpus,tmp_path_factory):
    out=tmp_path_factory.mktemp("training-single")
    subprocess.run(command(corpus,out),env=launch_env(),check=True,timeout=240)
    state=json.loads((out/"complete.json").read_text())
    assert state["step"]==2 and state["consumed"]==8 and state["threshold"] is not None
    table=Corpus(corpus/"train").table
    expected=consumption_order(table,17,8)[:8]
    snapshot=torch.load(out/"latest.pt",weights_only=False)
    assert expected.tolist()==snapshot["consumed_indices"].tolist()
    perf=json.loads((out/"performance.json").read_text())
    expected_rows=sum(Corpus(corpus/"train")[int(i)]["values"].shape[0] for i in expected)
    assert perf["training_rows"]==expected_rows and perf["training_windows"]==8
    assert perf["training_seconds"]>0 and perf["training_gpu_hours"]==0
    assert perf["training_real_rows_per_second"]==pytest.approx(expected_rows/perf["training_seconds"])
    ledger=json.loads((out/"parameter-ledger.json").read_text())
    assert sum(ledger["direct_by_module"].values())==ledger["total"]
    digest=hashlib.sha256(b"".join(bytes(table[i]["id"])+b"\n" for i in expected)).hexdigest()
    assert state["consumed_id_sha256"]==digest and state["reconstructed_prefix"]==0
    return out


def test_resume_is_bit_exact(corpus,single,tmp_path):
    snapshot=single/"step-000001.pt"
    for name in ("latest.pt","best.pt"):
        shutil.copyfile(snapshot,tmp_path/name)
    subprocess.run(command(corpus,tmp_path)+["--resume"],env=launch_env(),check=True,timeout=240)
    a=torch.load(single/"latest.pt",weights_only=False)
    b=torch.load(tmp_path/"latest.pt",weights_only=False)
    assert a["cursor"]==b["cursor"]==8 and a["step"]==b["step"]==2
    assert a["resources"]["training_rows"]==b["resources"]["training_rows"]
    for name,tensor in a["model"].items():
        assert torch.equal(tensor,b["model"][name]),name
    for key in ("complete.json","consumption.json","synthetic-test.json"):
        assert json.loads((single/key).read_text())==json.loads((tmp_path/key).read_text())


def test_two_rank_single_rank_parity(corpus,single,tmp_path):
    # Use the same env:// rendezvous as torchrun. Launching ranks directly also
    # works on Windows wheels with a TCPStore but without the elastic libuv backend.
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1",0))
        port=str(listener.getsockname()[1])
    workers=[]
    log_handles=[]
    try:
        for rank in range(2):
            f=open(tmp_path/f"rank-{rank}.log","wb")
            log_handles.append(f)
            env={**launch_env(),"MASTER_ADDR":"127.0.0.1","MASTER_PORT":port,
                 "WORLD_SIZE":"2","RANK":str(rank),"LOCAL_RANK":str(rank)}
            workers.append(subprocess.Popen(command(corpus,tmp_path),env=env,stdout=f,stderr=f))
        for rank,worker in enumerate(workers):
            status=worker.wait(timeout=150)
            assert status==0,(tmp_path/f"rank-{rank}.log").read_text(errors="replace")[-3000:]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait()
        for f in log_handles:
            f.close()
    a=torch.load(single/"latest.pt",weights_only=False)
    b=torch.load(tmp_path/"latest.pt",weights_only=False)
    assert b["contract"]["world"]==2 and len(b["rng"])==2
    assert a["selection"]["denominators"]==b["selection"]["denominators"]
    assert a["step"]==b["step"]==2 and a["cursor"]==b["cursor"]==8
    assert a["resources"]["training_rows"]==b["resources"]["training_rows"]
    for key in ("complete.json","consumption.json"):
        first=json.loads((single/key).read_text())
        second=json.loads((tmp_path/key).read_text())
        if key=="complete.json":
            assert first["consumed_id_sha256"]==second["consumed_id_sha256"]
            assert first["root_map"]==pytest.approx(second["root_map"],abs=2e-4)
            assert first["point_ap"]==pytest.approx(second["point_ap"],abs=2e-4)
        else:
            assert first==second
    # Null-gradient directions can differ slightly after Adam's epsilon scaling.
    for name,tensor in a["model"].items():
        torch.testing.assert_close(tensor,b["model"][name],rtol=1e-4,atol=1e-4,msg=name)
