"""Leakage-controlled normalization and tiled inference. Measurements use [time,channel]."""
from pathlib import Path
import numpy as np
import torch


def starts(length, width, stride):
    if length <= 0 or not 0 < stride <= width:
        raise ValueError("expected a nonempty series and 0 < stride <= width")
    last = max(0, length - width)
    indices = list(range(0, last + 1, stride))
    if indices[-1] != last:
        indices.append(last)
    return indices


def normalizer(reference):
    center = np.zeros(reference.shape[1], dtype=np.float32)
    scale = np.ones_like(center)
    for j in range(len(center)):
        values = reference[:, j]
        values = values[np.isfinite(values)]
        if values.size:
            low, center[j], high = np.quantile(values, [0.25, 0.5, 0.75])
            if high - low >= 1e-3:
                scale[j] = max(1e-3, (high - low) / 1.349)
    return center, scale


class Predictor:
    def __init__(self, path, device):
        from shifted_attention.config import ModelConfig
        from shifted_attention.model import ShiftedAttentionModel
        self.device = torch.device(device)
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = dict(ckpt["model_config"])
        weights = ckpt["model"]
        if config.get("aux_reconstruction"):
            config["aux_reconstruction"] = False
            weights = {k:v for k,v in weights.items() if k != "mask_embedding" and not k.startswith("reconstructor.")}
        self.model = ShiftedAttentionModel(ModelConfig(**config))
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def window(self, series, time, stats):
        center, scale = stats
        valid = np.isfinite(series)
        values = np.where(valid, np.clip((series - center) / scale, -30, 30), 0)
        def tensor(x, dtype):
            return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=self.device).unsqueeze(0)
        result = self.model(tensor(values, torch.float32), tensor(valid, torch.bool),
                            tensor(time - time[0], torch.float64),
                            tensor(np.ones(series.shape[1], dtype=bool), torch.bool))
        return {k: v[0].detach().cpu().numpy() for k, v in result.items()
                if k in ("root_logits", "root_valid", "anomaly_logits", "anomaly_valid")}

    def anomaly(self, series, time, first=0, ref=None, width=2048, stride=1024):
        total = np.zeros(len(series) - first, dtype=np.float64)
        hits = np.zeros(len(total), dtype=np.int32)
        fixed = None if ref is None else normalizer(ref)
        for rel in starts(len(total), width, stride):
            a = first + rel
            b = min(len(series), a + width)
            stats = fixed if fixed is not None else normalizer(series[max(0, a-512):a])
            out = self.window(series[a:b], time[a:b], stats)
            valid = out["anomaly_valid"].reshape(-1)
            logits = out["anomaly_logits"].reshape(-1)
            if not np.isfinite(logits[valid]).all():
                raise ValueError("nonfinite valid anomaly logits")
            total[rel:rel + b-a] += np.where(valid, logits, 0)
            hits[rel:rel + b-a] += valid
        mean = np.divide(total, hits, where=hits>0, out=np.zeros_like(total))
        return np.where(hits>0, 1/(1+np.exp(-np.clip(mean,-700,700))), 0), hits

    def root(self, series, time, a, b, ref=None):
        stats = normalizer(ref if ref is not None else series[max(0,a-512):a])
        score = np.zeros(series.shape[1])
        evidence = np.zeros(series.shape[1], dtype=bool)
        indices = starts(b-a, 512, 128)
        for rel in indices:
            begin, end = a + rel, min(b, a + rel + 512)
            out = self.window(series[begin:end], time[begin:end], stats)
            logits = out["root_logits"].reshape(-1).astype(np.float64)
            valid = out["root_valid"].reshape(-1)
            if not np.isfinite(logits[valid]).all():
                raise ValueError("nonfinite valid root logits")
            score = np.maximum(score, np.where(valid, 1/(1+np.exp(-np.clip(logits,-700,700))), 0))
            evidence |= valid
        return score, evidence, len(indices)
