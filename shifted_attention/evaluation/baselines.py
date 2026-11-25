"""Reference detectors operating on the same retained rows and event candidates."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from .assets import AUTHOR_WEIGHTS, TIMRCD_HUB_REV, digest, metric_source, save_json
from .inference import normalizer, starts
from .author_attention import install as install_author_attention


class Severity:
    """Max absolute robust deviation; use a strictly increasing bounded score for metric interfaces."""
    identity = {"predictor":"severity","formula":"z=clip(abs(x-med)/robust_scale,0,30); score=z/(1+z)"}

    @staticmethod
    def _scores(x,stats):
        mid,den = stats
        valid = np.isfinite(x)
        z = np.where(valid,np.minimum(np.abs(x-mid)/den,30),0)
        return z/(1+z),valid

    def root(self,series,time,a,b,ref=None):
        stats = normalizer(ref if ref is not None else series[max(0,a-512):a])
        values,valid = self._scores(series[a:b],stats)
        return values.max(axis=0),valid.any(axis=0),len(starts(b-a,512,128))

    def anomaly(self,series,time,first=0,ref=None,width=2048,stride=1024):
        fixed = normalizer(ref) if ref is not None else None
        total = np.zeros(len(series)-first)
        hits = np.zeros(len(total),dtype=np.int32)
        for k in starts(len(total),width,stride):
            a,b = first+k,min(len(series),first+k+width)
            stats = fixed if fixed is not None else normalizer(series[max(0,a-512):a])
            vals,valid = self._scores(series[a:b],stats)
            total[k:k+b-a] += vals.max(axis=1)
            hits[k:k+b-a] += valid.any(axis=1)
        return np.divide(total,hits,where=hits>0,out=np.zeros_like(total)),hits


def tsb_state_names(state):
    """The TSB port spells the eight author multi-head bias tables 'embedding' instead of 'emd'."""
    expected = {f"ts_encoder.transformer_encoder.layers.{i}.self_attn.binary_attention_bias.emd.weight"
                for i in range(8)}
    seen = {k.removeprefix("module.") for k in state if k.endswith("binary_attention_bias.emd.weight")}
    if seen!=expected:
        raise ValueError("Unexpected keys for the fixed published multivariate TimeRCD weights")
    converted, aliases = {},{}
    for key, value in state.items():
        new = key.replace("binary_attention_bias.emd.weight","binary_attention_bias.embedding.weight")
        if new in converted or (new!=key and new in state):
            raise ValueError("Non-bijective TSB state dictionary")
        converted[new] = value
        if new!=key:
            aliases[key] = new
    return converted,aliases


def tsb_multi_checkpoint(path):
    source = Path(path)
    derived = source.with_name("tsb-multi-strict.pth")
    manifest = derived.with_suffix(".json")
    if derived.exists() and manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf8"))
        if meta.get("source_sha256")==AUTHOR_WEIGHTS["multi"] and digest(derived)==meta.get("derived_sha256"):
            return str(derived)
    original = torch.load(source,map_location="cpu",weights_only=True)
    state = original.get("model_state_dict",original)
    converted,aliases = tsb_state_names(state)
    with tempfile.NamedTemporaryFile(dir=source.parent,delete=False) as f:
        torch.save(converted,f)
    tmp = Path(f.name)
    disk = torch.load(tmp,map_location="cpu",weights_only=True)
    if set(disk)!=set(converted) or any(not torch.equal(value,disk[aliases.get(key,key)]) for key,value in state.items()):
        tmp.unlink()
        raise ValueError("Tensor equality check failed for the TSB multivariate state")
    checksum = digest(tmp)
    os.replace(tmp,derived)
    save_json(manifest,{"source_sha256":AUTHOR_WEIGHTS["multi"],"derived_sha256":checksum,
                        "aliases":aliases,"identical_tensors_checked":len(state)})
    return str(derived)


class AuthorTimeRCD:
    def __init__(self,base,device,paper,width,query_rows=128):
        self.device, self.paper, self.width = device,paper,width
        self.query_rows = query_rows
        self.paths = {}
        for variant,wanted in AUTHOR_WEIGHTS.items():
            path = base/"author-weights"/f"pretrain_checkpoint_best_{variant}.pth"
            if not path.is_file() or digest(path)!=wanted:
                raise ValueError(f"Expected published {variant} checkpoint (prepare --assets author)")
            self.paths[variant] = str(path.resolve())
        if paper:
            source = metric_source(base,"Original")
            sys.path.insert(0,str(source.resolve()))
            self.detector = importlib.import_module("time_rcd").TimeRCDDetector
        else:
            source = metric_source(base,"TSBModel")
            sys.path.insert(0,str(source.resolve()))
            self.dispatch = importlib.import_module("TSB_AD.model_wrapper").run_Time_RCD
            self.tsb_class = importlib.import_module("TSB_AD.models.Time_RCD").Time_RCD
            self.paths["multi"] = tsb_multi_checkpoint(self.paths["multi"])
        self.last_variant = None
        self.instance = None
        self.identity = {"predictor":"TimeRCD-authors","hf_revision":TIMRCD_HUB_REV,
                         "checkpoint_sha256_by_variant":AUTHOR_WEIGHTS,
                         "inference_commit":"372bb980426b2f67007311c6f3165ab789c79bef" if paper else
                         "303dffbfc11612995b17b06633f8c9d0ea0e38e5",
                         "batch_size":64,"normalization":"authors' entire scored input"}
        if not paper:
            self.identity["tsb_multi_namespace"] = json.loads(
                Path(self.paths["multi"]).with_suffix(".json").read_text(encoding="utf8"))

    def anomaly(self,series,time,first=0,ref=None,width=None,stride=None):
        x = series[first:]
        if not np.isfinite(x).all():
            raise ValueError("The official baseline expects retained finite rows")
        variant = "uni" if x.shape[1]==1 else "multi"
        physical_batch = 1 if variant=="multi" else 64
        layers = []
        if self.paper or variant=="multi":
            # The paper API and TSB dispatcher both construct this model from the intact full-width checkpoint.
            instance_key = (variant,x.shape[1])
            if instance_key!=self.last_variant:
                self.instance = None
                if self.paper:
                    self.instance = self.detector.from_local(
                        self.paths[variant],variant=variant,win_size=self.width,batch_size=physical_batch,device=self.device)
                else:
                    self.instance = self.tsb_class(win_size=self.width,input_c=x.shape[1],batch_size=physical_batch,
                                                   device=self.device,checkpoint=self.paths[variant])
                self.last_variant = instance_key
            if variant=="multi":
                if self.paper:
                    self.instance._ensure_tester(x.shape[1])
                    model = self.instance._tester.model
                else:
                    model = self.instance.model
                layers = install_author_attention(model,query_rows=self.query_rows)
            if self.paper:
                result = self.instance.predict(x)
            else:
                self.instance.zero_shot(x)
                result = self.instance.decision_scores_
        else:
            result = self.dispatch(x,win_size=self.width,batch_size=64,device=self.device,
                                   checkpoint=self.paths[variant])
        result = np.asarray(result,dtype=np.float64).reshape(-1)
        if len(result)!=len(x) or not np.isfinite(result).all():
            raise ValueError("Original TimeRCD output is misaligned/nonfinite")
        self.execution = {"physical_batch_size":physical_batch,"native_batch_setting":64,
                          "input_channels":x.shape[1],"input_rows":len(x),
                          "query_tiles":[{"full_keys":keys,"query_rows":rows}
                                         for keys,rows in sorted(set().union(*(l._eval_tiles for l in layers)))],
                          "dense_layer_calls":sum(l._eval_dense_calls for l in layers) if layers else None}
        return result,np.ones(len(x),dtype=np.int32)
