"""Reference detectors operating on the same retained rows and event candidates."""
import importlib
import sys

import numpy as np

from .assets import AUTHOR_WEIGHTS, TIMRCD_HUB_REV, digest, metric_source
from .inference import normalizer, starts


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


class AuthorTimeRCD:
    def __init__(self,base,device,paper,width):
        self.device, self.paper, self.width = device,paper,width
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
        self.last_variant = None
        self.instance = None
        self.identity = {"predictor":"TimeRCD-authors","hf_revision":TIMRCD_HUB_REV,
                         "checkpoint_sha256_by_variant":AUTHOR_WEIGHTS,
                         "inference_commit":"372bb980426b2f67007311c6f3165ab789c79bef" if paper else
                         "303dffbfc11612995b17b06633f8c9d0ea0e38e5",
                         "batch_size":64,"normalization":"authors' entire scored input"}

    def anomaly(self,series,time,first=0,ref=None,width=None,stride=None):
        x = series[first:]
        if not np.isfinite(x).all():
            raise ValueError("The official baseline expects retained finite rows")
        variant = "uni" if x.shape[1]==1 else "multi"
        if self.paper:
            if variant!=self.last_variant:
                self.instance = self.detector.from_local(
                    self.paths[variant],variant=variant,win_size=self.width,batch_size=64,device=self.device)
                self.last_variant = variant
            result = self.instance.predict(x)
        else:
            result = self.dispatch(x,win_size=self.width,batch_size=64,device=self.device,
                                   checkpoint=self.paths[variant])
        result = np.asarray(result,dtype=np.float64).reshape(-1)
        if len(result)!=len(x) or not np.isfinite(result).all():
            raise ValueError("Original TimeRCD output is misaligned/nonfinite")
        return result,np.ones(len(x),dtype=np.int32)
