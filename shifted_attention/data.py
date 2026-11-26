"""Read-only indexed access to the generator's public input and supervision streams."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

FAMILIES = ("linear", "additive", "interaction", "saturating")
MODES = ("healthy", "process", "observation", "mixed")
PROFILE = "sster-direct-state-20261125-v1"
ROUND1_EDITION = "round1-1680k-v1"
ROUND1_PER_WIDTH = 240_000
PRODUCTION_WIDTHS = (1,4,8,16,32,64,128)
DTYPE = np.dtype([("shard", "<u4"), ("io", "<u8"), ("il", "<u4"), ("to", "<u8"),
                  ("tl", "<u4"), ("n", "u1"), ("k", "u1"), ("mode","u1"), ("family","u1"),
                  ("ancestor","<u4"), ("focal","u1"), ("id","S32")])
INPUTS = ("values","observed","time","channel_valid")
TARGETS = ("Y","Y_known","R","R_known","E")


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.part")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    os.replace(temp, path)


def _release_info(split):
    release = json.loads((split/"_SUCCESS").read_text(encoding="utf8"))
    if release["profile"] != PROFILE:
        raise ValueError(f"unexpected corpus profile in {split}")
    raw_audit = (split/"audit.json").read_bytes()
    if hashlib.sha256(raw_audit).hexdigest() != release["audit_sha256"]:
        raise ValueError(f"release audit hash mismatch in {split}")
    audit = json.loads(raw_audit)
    edition = release.get("training_edition")
    if (("training_edition" in release)!=("training_edition" in audit) or edition != audit.get("training_edition")
            or ("training_edition" in release and edition!=ROUND1_EDITION)):
        raise ValueError(f"unrecognized or inconsistent training edition in {split}")
    role=audit.get("split")
    if edition is not None and (role!="train" or release["smoke"]):
        raise ValueError(f"training edition applies only to a formal train split: {split}")
    if not release["smoke"]:
        if role not in ("train","validation","test") or any(audit.get(k)!=release[k] for k in ("windows","manifest_sha256","profile","smoke")):
            raise ValueError(f"release marker and audit disagree in {split}")
        per_width = (ROUND1_PER_WIDTH if edition else 960_000) if role=="train" else 48_000
        totals=Counter()
        for n,k,mode,count in audit["cells"]:
            totals[n]+=count
        if (release["windows"]!=per_width*len(PRODUCTION_WIDTHS) or
                totals!={n:per_width for n in PRODUCTION_WIDTHS}):
            raise ValueError(f"unexpected corpus counts for the production profile in {split}")
    return release,audit


def production_edition(root):
    """Accept either full training edition only with compatible, authenticated held-out splits."""
    info=[_release_info(root/name) for name in ("train","validation","test")]
    if (any(release["smoke"] for release,_ in info) or
            [audit.get("split") for _,audit in info]!=["train","validation","test"]):
        raise ValueError("production training requires three formal split releases")
    calibrations=[audit["calibration_sha256"] for _,audit in info]
    if (any(not isinstance(sha,str) or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha)
            for sha in calibrations) or len(set(calibrations))!=1):
        raise ValueError("split calibration hashes differ")
    return info[0][0].get("training_edition")


def build_index(split):
    split = Path(split)
    source = split / "manifest.jsonl"
    release,_ = _release_info(split)
    edition = release.get("training_edition")
    st = source.stat()
    fingerprint = [st.st_size, st.st_mtime_ns, DTYPE.itemsize, release["manifest_sha256"], 3]
    if edition:
        fingerprint += [edition,release["audit_sha256"]]
    meta = split / "reader-index.json"
    if meta.exists():
        previous = json.loads(meta.read_text(encoding="utf8"))
        indexed = split / previous["data"]
        if (previous["fingerprint"] == fingerprint and previous["count"] == release["windows"]
                and previous["manifest_sha256"] == release["manifest_sha256"] and previous.get("training_edition")==edition
                and previous["data"] == "reader-"+release["manifest_sha256"]+".bin"
                and indexed.is_file() and indexed.stat().st_size == previous["count"] * DTYPE.itemsize):
            return
    shards = {}
    buffer = []
    digest = hashlib.sha256()
    widths = Counter()
    count = 0
    with tempfile.NamedTemporaryFile(dir=split, delete=False) as f:
        tmp = Path(f.name)
        try:
            with source.open("rb") as manifest:
                for line in manifest:
                    digest.update(line)
                    r = json.loads(line)
                    if edition:
                        widths[r["N"]]+=1
                    path = r["shard"]
                    if Path(path).is_absolute() or ".." in Path(path).parts:
                        raise ValueError("invalid shard path")
                    focus = r["focal"]
                    code = 0
                    if focus is not None:
                        layer, kind, _ = focus
                        if layer not in MODES[1:3] or kind[0] not in "LS" or not 1 <= int(kind[1:]) <= 20:
                            raise ValueError("invalid template code")
                        code = int(kind[1:]) + 20*(kind[0]=="S") + 40*(layer=="observation")
                    buffer.append((shards.setdefault(path,len(shards)), r["input_offset"],r["input_size"],
                                   r["target_offset"],r["target_size"],r["N"],r["K"],MODES.index(r["mode"]),
                                   FAMILIES.index(r["family"]),r["ancestor"],code,r["id"].encode("ascii")))
                    count += 1
                    if len(buffer) == 8192:
                        np.asarray(buffer,dtype=DTYPE).tofile(f)
                        buffer.clear()
                np.asarray(buffer,dtype=DTYPE).tofile(f)
            if not count:
                raise ValueError("empty corpus split")
            if count != release["windows"] or digest.hexdigest() != release["manifest_sha256"]:
                raise ValueError(f"manifest differs from the verified generator release in {split}")
            if edition and widths!={n:ROUND1_PER_WIDTH for n in PRODUCTION_WIDTHS}:
                raise ValueError("manifest channel-width counts disagree with the round-one release")
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    name = "reader-" + digest.hexdigest() + ".bin"
    os.replace(tmp, split/name)
    metadata=dict(count=count, data=name, manifest_sha256=digest.hexdigest(),shards=list(shards),fingerprint=fingerprint)
    if edition:
        metadata["training_edition"]=edition
    atomic_json(meta,metadata)


class Corpus(Dataset):
    def __init__(self, split):
        self.split = Path(split)
        self.meta = json.loads((self.split/"reader-index.json").read_text(encoding="utf8"))
        release = json.loads((self.split/"_SUCCESS").read_text(encoding="utf8"))
        if (self.meta["count"]!=release["windows"] or self.meta["manifest_sha256"]!=release["manifest_sha256"] or
                self.meta.get("training_edition")!=release.get("training_edition") or
                self.meta["data"]!="reader-"+release["manifest_sha256"]+".bin"):
            raise ValueError(f"reader index does not belong to the selected release in {self.split}")
        self.table = np.memmap(self.split/self.meta["data"], mode="r", dtype=DTYPE, shape=(self.meta["count"],))
        self._handles = OrderedDict()
        self._pid = None

    def __len__(self):
        return len(self.table)

    def __getstate__(self):
        return {"split": self.split}

    def __setstate__(self, state):
        self.__init__(state["split"])

    def _handle(self, shard):
        if self._pid != os.getpid():
            for h in self._handles.values():
                h.close()
            self._handles.clear()
            self._pid = os.getpid()
        if shard in self._handles:
            self._handles.move_to_end(shard)
        else:
            self._handles[shard] = (self.split/self.meta["shards"][shard]).open("rb")
            if len(self._handles)>16:
                self._handles.popitem(last=False)[1].close()
        return self._handles[shard]

    def __getitem__(self, index):
        r = self.table[index]
        handle = self._handle(int(r["shard"]))
        sample = {}
        for offset, length, keys in (("io","il",INPUTS), ("to","tl",TARGETS)):
            handle.seek(int(r[offset]))
            payload = handle.read(int(r[length]))
            if len(payload)!=int(r[length]):
                raise IOError(f"truncated record {r['id']!r}")
            with np.load(io.BytesIO(payload),allow_pickle=False) as data:
                sample.update({key:torch.from_numpy(data[key]) for key in keys})
        sample["index"] = int(index)
        return sample


def collate(samples):
    batch = len(samples)
    length = max(s["values"].shape[0] for s in samples)
    width = max(s["values"].shape[1] for s in samples)
    result = {k:torch.zeros((batch, length, width),dtype=dtype) for k,dtype in
              (("values",torch.float32),("observed",torch.bool))}
    for key in ("time","row_valid","Y","Y_known"):
        result[key] = torch.zeros((batch,length),dtype=torch.float64 if key=="time" else torch.bool)
    for key in ("channel_valid","R","R_known","E"):
        result[key] = torch.zeros((batch,width),dtype=torch.bool)
    for j,s in enumerate(samples):
        t,c = s["values"].shape
        result["row_valid"][j,:t] = True
        for key in ("values","observed"):
            result[key][j,:t,:c] = s[key]
        for key in ("time","Y","Y_known"):
            result[key][j,:t] = s[key]
        for key in ("channel_valid","R","R_known","E"):
            result[key][j,:c] = s[key]
    result["index"] = torch.tensor([s["index"] for s in samples])
    return result


def apportion(counts, seats):
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    base, remainder = np.divmod(counts*seats,total)
    order = np.argsort(-remainder,kind="stable")
    base[order[:seats-int(base.sum())]] += 1
    return base


def consumption_order(table, seed, budget, epoch=0):
    """C.3 nested largest-remainder allocation, then independent random permutations."""
    if not 0 < budget <= len(table):
        raise ValueError("budget must fall inside an epoch")
    rng = np.random.default_rng(np.random.SeedSequence([seed,epoch]))
    strata = ((table["n"].astype(np.int32)*7+table["k"])*4+table["mode"])*128+table["focal"]
    ordered = np.argsort(strata,kind="stable")
    keys, sizes = np.unique(strata[ordered],return_counts=True)
    starts = np.r_[0,sizes.cumsum()]
    main = keys//128
    cells, boundaries = np.unique(main,return_index=True)
    ends = np.r_[boundaries[1:],len(keys)]
    quotas = apportion(np.add.reduceat(sizes,boundaries),budget)
    front,back=[],[]
    for left,right,quota in zip(boundaries,ends,quotas):
        for idx,take in enumerate(apportion(sizes[left:right],int(quota)),int(left)):
            block = ordered[starts[idx]:starts[idx+1]].copy()
            rng.shuffle(block)
            front.append(block[:take])
            back.append(block[take:])
    front,back = np.concatenate(front),np.concatenate(back)
    rng.shuffle(front)
    rng.shuffle(back)
    return np.concatenate((front,back))


def validation_halves(table, tiny=False):
    keys = np.stack([table["n"],table["family"],table["ancestor"]],axis=1)
    unique, inv = np.unique(keys,axis=0,return_inverse=True)
    buckets = defaultdict(list)
    half = np.zeros(len(unique),dtype=bool)
    for j,(n,f,i) in enumerate(unique):
        full = f"{PROFILE}|generic|{n}|{FAMILIES[f]}|{i}"
        buckets[(n,f)].append((hashlib.sha256(full.encode()).hexdigest(),int(i),j))
    for (n,f), group in buckets.items():
        if not tiny and len(group)!=750:
            raise ValueError(f"validation ancestor count for {(n,f)}: {len(group)}")
        take = (int(f)%2 == 0) if len(group)==1 else len(group)//2
        for _,_,j in sorted(group)[:int(take)]:
            half[j] = True
    mask = half[inv.reshape(-1)]
    return np.flatnonzero(mask),np.flatnonzero(~mask)
