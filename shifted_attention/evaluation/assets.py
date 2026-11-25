"""Acquire released benchmark inputs into a user-selected directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import requests

TIMERCD_REV = "768bb4e2b019280bd47eb116ef976ca1b8360b0f"
TSB_REV = "303dffbfc11612995b17b06633f8c9d0ea0e38e5"
TSB_ARCHIVE_SHA256 = {"U":"0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961",
                      "M":"7de86ac27f30eeb48d833bb061055670e3f3de07defd995cf2bd5db10ccc9a0d"}
AERCA_REV = "c3dd49d88568d77c3e1ecb9a40472cd77a79b930"
TIMRCD_HUB_REV = "0880420070a524efe14d7e163b34dcca90d2be1b"
AUTHOR_WEIGHTS = {"uni":"be8906c6a62f16cedd8fe0fe48ef11c4363099633ed6ba8780af819337f7ffaa",
                  "multi":"a497ac910f24f8fb347d690a21b60d74bc04785ebc777880553c158dea4434d0"}
CHAMBERS = {
    "lt_interventions_standard_v1": "476664d024f88e8b7640998bb5e9ee33",
    "wt_intake_impulse_v1": "8b7a0e587d3e9a804b380d96f14a1b12",
}


def digest(path, algo="sha256"):
    h = hashlib.new(algo)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent, delete=False) as f:
        json.dump(obj, f, allow_nan=False, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(f.name, path)


def download(url, path, checksum=None, algo="sha256"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with requests.get(url, stream=True, timeout=(15, 180)) as r:
            r.raise_for_status()
            if r.status_code != 200:
                raise OSError(f"Incomplete HTTP response {r.status_code}: {url}")
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                try:
                    for chunk in r.iter_content(2**20):
                        f.write(chunk)
                except BaseException:
                    f.close()
                    Path(f.name).unlink(missing_ok=True)
                    raise
        tmp = Path(f.name)
        if checksum and digest(tmp, algo) != checksum:
            tmp.unlink()
            raise ValueError(f"Checksum mismatch: {url}")
        os.replace(tmp, path)
    if checksum and digest(path, algo) != checksum:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def unzip_csv(archive, folder):
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = set()
        for member in z.infolist():
            parts = Path(member.filename.replace("\\", "/")).parts
            name = parts[-1]
            if not name.endswith(".csv") or any(s.startswith((".", "__MACOSX")) for s in parts):
                continue
            if name in names:
                raise ValueError(f"Ambiguous CSV basename {name} in {archive}")
            names.add(name)
            with z.open(member) as source, tempfile.NamedTemporaryFile(dir=folder,delete=False) as out:
                shutil.copyfileobj(source, out)
            os.replace(out.name, folder/name)
    if not names:
        raise ValueError(f"No CSV data in {archive}")


def git_blob(body):
    return hashlib.sha1(b"blob "+str(len(body)).encode()+b"\0"+body).hexdigest()


def metric_source(base, name, fetch=False):
    """Lock values originate in git ls-tree on the exact commits and are distributed with this package."""
    registry = json.loads(Path(__file__).with_name(name+"_lock.json").read_text(encoding="utf8"))
    repo, commit = registry["repo"],registry["commit"]
    root = base/"source-verified"/("TSB" if name=="TSBModel" else name)
    for rel, expected in registry["files"].items():
        path = root/rel
        if path.is_file() and git_blob(path.read_bytes())==expected:
            continue
        if not fetch:
            asset = "author" if name in ("Original","TSBModel") else "metrics"
            raise ValueError(f"Missing or incorrect Git blob {rel}; run prepare --assets {asset}")
        urls = (f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}",
                f"https://cdn.jsdelivr.net/gh/{repo}@{commit}/{rel}")
        errors = []
        for url in urls:
            try:
                response = requests.get(url,timeout=(12,40))
                response.raise_for_status()
                body = response.content
                if git_blob(body)!=expected:
                    raise ValueError("Git blob mismatch")
            except (requests.RequestException, ValueError) as exc:
                errors.append(str(exc))
                continue
            path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
                f.write(body)
            os.replace(f.name,path)
            break
        else:
            raise OSError(f"Failed to verify {repo}@{commit}/{rel}: {errors}")
    return root


def prepare(base, assets):
    base = Path(base)
    for asset in assets:
        if asset == "tsad":
            for split in ("U","M"):
                name = f"TSB-AD-{split}"
                url = "https://www.thedatum.org/datasets/"+name+".zip"
                path = download(url, base/"archives"/(name+".zip"), TSB_ARCHIVE_SHA256[split])
                unzip_csv(path, base/name)
                save_json(base/"provenance"/(name+".json"), {"url":url,"sha256":digest(path)})
        if asset in ("tsad","metrics"):
            for name in ("TimeRCD","TSB"):
                metric_source(base,name,fetch=True)
        elif asset == "chamber":
            for name, md5 in CHAMBERS.items():
                url = "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/"+name+".zip"
                path = download(url, base/"archives"/(name+".zip"), md5, "md5")
                unzip_csv(path, base/name)
                save_json(base/"provenance"/(name+".json"), {"url":url,"md5":md5,"sha256":digest(path)})
            from .adapters import chamber
            chamber(base)
        elif asset == "msds":
            from .adapters import msds
            msds(base)
            from .msds_services import prepare_service
            prepare_service(base)
        elif asset == "msds-services":
            from .msds_services import prepare_service
            prepare_service(base)
        elif asset == "author":
            for name in ("Original","TSB","TSBModel"):
                metric_source(base,name,fetch=True)
            for variant, checksum in AUTHOR_WEIGHTS.items():
                name = "pretrain_checkpoint_best_"+variant+".pth"
                url = f"https://huggingface.co/thu-sail-lab/Time-RCD/resolve/{TIMRCD_HUB_REV}/best_model/{name}"
                download(url,base/"author-weights"/name,checksum)
