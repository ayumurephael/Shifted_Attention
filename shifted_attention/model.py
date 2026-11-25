"""Dual-stream Transformer with learnable shifted multi-head channel attention."""
import math
import time as wallclock

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttention, ShiftedChannelAttention, StateLinear, masked_rows
from .config import ModelConfig


class CorrectedProjection(nn.Module):
    """An affine map plus an initially zero learned nonlinear correction."""
    def __init__(self, in_features, out_features, linear_only=False):
        super().__init__()
        self.skip = StateLinear(in_features, out_features)
        self.hidden = None if linear_only else StateLinear(in_features, out_features)
        self.correction = None if linear_only else StateLinear(out_features, out_features)

    def forward(self, x):
        base = self.skip(x)
        if self.correction is None:
            return base
        return base + self.correction(F.gelu(self.hidden(x)))


def feed_forward(dim, bias=True):
    return nn.Sequential(StateLinear(dim, dim * 4, bias=bias), nn.GELU(),
                         StateLinear(dim * 4, dim, bias=bias))


def patch_positions(time, row_valid, patch_width, dim):
    # FP64 subtraction before reduction prevents loss of precision at large epochs.
    clock = time.double()
    delta = clock[:, 1:] - clock[:, :-1]
    good = row_valid[:, 1:] & row_valid[:, :-1] & (delta > 0)
    if clock.shape[1] == 1:
        step = torch.ones_like(clock[:, 0])
    else:
        ordered = torch.sort(torch.where(good, delta, torch.inf), dim=-1).values
        count = good.sum(dim=-1, keepdim=True)
        lo = ordered.gather(1, ((count - 1).clamp_min(0) // 2))
        hi = ordered.gather(1, (count // 2))
        step = torch.where(count > 0, (lo + hi) / 2, 1.0).squeeze(1)
    relative = torch.where(row_valid, clock - clock[:, :1], 0.0)
    patch_time = relative.unflatten(-1, (-1, patch_width)).sum(dim=-1)
    n = row_valid.unflatten(-1, (-1, patch_width)).sum(dim=-1).clamp_min(1)
    pos = patch_time / (n * patch_width * step[:, None])
    k = torch.arange(0, dim, 2, device=time.device, dtype=torch.float64)
    phase = pos[:, :, None] * torch.exp(k * (-math.log(10000.0) / dim))
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class DualStreamBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        s = cfg.stream_width
        self.time_norm = nn.LayerNorm(s)
        self.time_attention = MultiHeadAttention(s, cfg.heads)
        self.channel_norm = nn.LayerNorm(s)
        self.channel_attention = ShiftedChannelAttention(cfg)
        self.t_ff_norm = nn.LayerNorm(s)
        self.c_ff_norm = nn.LayerNorm(s, bias=False)
        self.t_ff = feed_forward(s)
        self.c_ff = feed_forward(s, bias=False)
        self.feedback = cfg.variant == "feedback"

    def forward(self, temporal, relational, valid):
        time_input = temporal + relational if self.feedback else temporal
        u = self.time_norm(time_input).flatten(0, 1)
        m = valid.flatten(0, 1)
        a = masked_rows(time_input + self.time_attention(u, u, m, m).reshape_as(time_input), valid)
        b = masked_rows(relational + self.channel_attention(self.channel_norm(a), valid), valid)
        return masked_rows(a + self.t_ff(self.t_ff_norm(a)), valid), masked_rows(b + self.c_ff(self.c_ff_norm(b)), valid)


class FlatBlock(nn.Module):
    """Data- and readout-matched any-variate control, with same/different biases."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg.final_width
        self.norm = nn.LayerNorm(d)
        self.attention = MultiHeadAttention(d, cfg.heads)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = feed_forward(d)
        self.channel_bias = nn.Parameter(torch.zeros(cfg.heads, 2))
        self.chunk = cfg.source_chunk_size

    def forward(self, x, valid):
        b, c, p, d = x.shape
        m = valid.flatten(1, 2)
        z = self.norm(x).reshape(b, c*p, d)
        channel_ids = torch.arange(c, device=x.device).repeat_interleave(p)
        att = self.attention(z, z, m, m, self.chunk, channel_ids, self.channel_bias)
        u = masked_rows(x + att.reshape_as(x), valid)
        return masked_rows(u + self.ff(self.ff_norm(u)), valid)


class SeedPool(nn.Module):
    """Shared learned seeds; the whole residual path vanishes on empty sets."""
    def __init__(self, dim, heads, seeds, source_chunk=None):
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(seeds, dim))
        nn.init.normal_(self.seeds, std=0.02)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, heads)
        self.ff = feed_forward(dim)
        self.source_chunk = source_chunk

    def forward(self, memory, valid):
        seed = self.seeds.unsqueeze(0).expand(memory.shape[0], -1, -1)
        # Match residual dtype to the surrounding features under mixed precision.
        seed = seed.to(memory.dtype)
        query_ok = valid.any(dim=-1, keepdim=True).expand(-1, seed.shape[1])
        a = seed + self.attention(self.q_norm(seed), self.kv_norm(memory), valid,
                                  query_ok, self.source_chunk)
        return masked_rows(a + self.ff(self.ff_norm(a)), query_ok)


class ShiftedAttentionModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = cfg = config or ModelConfig()
        w, s, d = cfg.patch_width, cfg.stream_width, cfg.final_width
        self.flat = cfg.variant == "flat"
        self.tokenizer = CorrectedProjection(3*w, d if self.flat else s, cfg.linear_tokenizer)
        self.blocks = nn.ModuleList((FlatBlock(cfg) if self.flat else DualStreamBlock(cfg))
                                    for _ in range(cfg.blocks))
        self.t_final = nn.LayerNorm(d if self.flat else s)
        if not self.flat:
            self.c_final = nn.LayerNorm(s, bias=False)
        self.root_pool = SeedPool(d, cfg.heads, cfg.root_seeds)
        self.root_compress = CorrectedProjection(cfg.root_seeds*d, d)
        self.root_norm = nn.LayerNorm(d)
        self.root_self = MultiHeadAttention(d, cfg.heads)
        self.root_score = nn.Sequential(nn.LayerNorm(d), StateLinear(d, 1))
        self.unpatch = StateLinear(d, d if cfg.repeat_patch_points else w*d)
        self.point_pool = SeedPool(d, cfg.heads, cfg.point_seeds, cfg.source_chunk_size)
        self.point_compress = CorrectedProjection(cfg.point_seeds*d, d)
        self.point_score = nn.Sequential(nn.LayerNorm(d), StateLinear(d, 1))
        if cfg.aux_reconstruction:
            self.mask_embedding = nn.Parameter(torch.empty(d if self.flat else s))
            nn.init.normal_(self.mask_embedding,std=.02)
            self.reconstructor = nn.Sequential(StateLinear(d,d), nn.GELU(), StateLinear(d,d),
                                              nn.GELU(), StateLinear(d,1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, CorrectedProjection) and m.correction is not None:
                nn.init.zeros_(m.correction.weight)
                nn.init.zeros_(m.correction.bias)

    def decay_parameter_groups(self, weight_decay):
        """Decoupled decay applies only to affine projection/FFN weights."""
        decay_ids = {id(m.weight) for m in self.modules() if isinstance(m, nn.Linear)}
        return [
            {"params": [p for p in self.parameters() if p.requires_grad and id(p) in decay_ids],
             "weight_decay": weight_decay},
            {"params": [p for p in self.parameters() if p.requires_grad and id(p) not in decay_ids],
             "weight_decay": 0.0},
        ]

    def tokenize(self, values, observed, time, channel_valid, row_valid=None):
        if values.ndim != 3 or min(values.shape) < 1 or observed.shape != values.shape:
            raise ValueError("Expected nonempty values and observed tensors [batch,time,channel].")
        b, t, c = values.shape
        if time.shape != (b,t) or channel_valid.shape != (b,c):
            raise ValueError("Time and channel masks must match observation axes.")
        if row_valid is None:
            row_valid = torch.ones((b, t), dtype=torch.bool, device=values.device)
        if row_valid.shape != (b,t):
            raise ValueError("row_valid must have shape [batch,time].")
        w = self.config.patch_width
        pad = (-t) % w
        rows = F.pad(row_valid.bool(), (0,pad))
        obs = observed.bool() & channel_valid.bool()[:, None, :] & row_valid.bool()[:, :, None]
        clean = torch.where(obs, values, 0.0)
        x_patch = F.pad(clean.transpose(1, 2), (0,pad)).unflatten(-1, (-1,w))
        observed_padded = F.pad(obs.transpose(1, 2), (0,pad))
        obs_patch = observed_padded.unflatten(-1, (-1,w))
        valid = obs_patch.any(dim=-1)
        row_patch = rows.unflatten(-1, (-1,w))[:,None].expand_as(obs_patch)
        patch_input = torch.cat((x_patch, obs_patch.to(clean.dtype), row_patch.to(clean.dtype)), dim=-1)
        z = self.tokenizer(patch_input)
        pos = patch_positions(F.pad(time, (0,pad)), rows, w, z.shape[-1])
        return masked_rows(z + pos[:, None].to(z.dtype), valid), valid, observed_padded

    def _run(self, fn, *args, rematerialize):
        if rematerialize and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def encode(self, tokens, valid):
        if self.flat:
            for block in self.blocks:
                tokens = self._run(block, tokens, valid, rematerialize=self.config.checkpoint_blocks)
            return masked_rows(self.t_final(tokens), valid)
        temporal, relational = tokens, torch.zeros_like(tokens)
        for block in self.blocks:
            temporal, relational = self._run(block, temporal, relational, valid,
                                              rematerialize=self.config.checkpoint_blocks)
        return masked_rows(torch.cat((self.t_final(temporal), self.c_final(relational)), dim=-1), valid)

    def root_head(self, encoded, valid):
        b, c, p, d = encoded.shape
        present = valid.any(dim=-1)
        pooled = self.root_pool(encoded.reshape(b*c, p, d), valid.reshape(b*c, p))
        u = masked_rows(self.root_compress(pooled.reshape(b,c,-1)), present)
        z = self.root_norm(u)
        summary = u + self.root_self(z,z,present,present,self.config.source_chunk_size)
        logits = self.root_score(summary).squeeze(-1)
        return torch.where(present, logits, 0.0), present

    def _point_chunk(self, encoded, observed):
        b,c,p,d = encoded.shape
        w = self.config.patch_width
        projected = self.unpatch(encoded)
        if self.config.repeat_patch_points:
            points = projected.unsqueeze(-2).expand(b,c,p,w,d)
        else:
            points = projected.reshape(b,c,p,w,d)
        features = points.reshape(b,c,p*w,d).transpose(1,2).reshape(b*p*w,c,d)
        valid = observed.transpose(1,2).reshape(b*p*w,c)
        pooled = self.point_pool(features, valid)
        summarised = self.point_compress(pooled.flatten(1))
        logits = self.point_score(summarised).reshape(b,p*w)
        return torch.where(valid.any(-1).reshape(b,p*w), logits, 0.0)

    def _reconstruction_sum(self,tokens,valid,selected,values,observed_pad,time,row_valid):
        if selected.shape != valid.shape or selected.dtype != torch.bool:
            raise ValueError("reconstruction mask must be boolean [B,C,P]")
        selected = selected & valid
        w, d = self.config.patch_width, self.config.final_width
        pad = (-values.shape[1]) % w
        rows = torch.ones_like(time,dtype=torch.bool) if row_valid is None else row_valid
        pos = patch_positions(F.pad(time,(0,pad)),F.pad(rows,(0,pad)),w,tokens.shape[-1])
        masked = (pos[:,None].to(tokens.dtype) + self.mask_embedding.to(tokens.dtype))
        memory = self.encode(torch.where(selected[...,None],masked,tokens),valid)[selected]
        observed = observed_pad.unflatten(-1,(-1,w))[selected]
        target = F.pad(values.transpose(1,2),(0,pad)).unflatten(-1,(-1,w))[selected]
        def reconstruct(x,y,measured):
            patch = self.unpatch(x)
            points = (patch[:,None].expand(-1,w,d) if self.config.repeat_patch_points
                      else patch.reshape(-1,w,d))
            prediction = self.reconstructor(points).squeeze(-1).float()
            residual = prediction - torch.where(measured,y,0.0).float()
            return torch.where(measured,residual.square(),0.).sum()
        stride = self.config.patch_chunk_size * valid.shape[1]
        # A zero-length slice still connects all auxiliary parameters to DDP.
        terms = [self._run(reconstruct,memory[i:i+stride],target[i:i+stride],observed[i:i+stride],
                           rematerialize=self.config.checkpoint_decoder)
                 for i in range(0,max(1,len(memory)),stride)]
        return torch.stack(terms).sum()

    def forward(self, values, observed, time, channel_valid, row_valid=None, reconstruction_mask=None):
        """Return logits and validity. Invalid logits are neutral placeholders.

        For scores use where(valid, sigmoid(logits), 0), preserving all benchmark
        rows and eligible candidates. time is the physical clock, not a row index.
        """
        tokens, valid, observed_pad = self.tokenize(values, observed, time, channel_valid, row_valid)
        encoded = self.encode(tokens, valid)
        root_logits, root_valid = self.root_head(encoded, valid)
        w, chunk = self.config.patch_width, self.config.patch_chunk_size
        point_scores = []
        for start in range(0, encoded.shape[2], chunk):
            stop = min(start + chunk, encoded.shape[2])
            point_scores.append(self._run(self._point_chunk, encoded[:,:,start:stop],
                                          observed_pad[:,:,start*w:stop*w],
                                          rematerialize=self.config.checkpoint_decoder))
        t = values.shape[1]
        output = {"root_logits": root_logits, "root_valid": root_valid,
                "anomaly_logits": torch.cat(point_scores, dim=-1)[:, :t],
                "anomaly_valid": observed_pad.any(dim=1)[:, :t]}
        if reconstruction_mask is not None:
            if not self.config.aux_reconstruction:
                raise ValueError("the reconstruction control must be enabled in ModelConfig")
            if values.is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                started = wallclock.perf_counter()
            output["reconstruction_sse"] = self._reconstruction_sum(
                tokens,valid,reconstruction_mask,values,observed_pad,time,row_valid)
            if values.is_cuda:
                end_event.record()
                end_event.synchronize()
                seconds = start_event.elapsed_time(end_event) / 1000
            else:
                seconds = wallclock.perf_counter() - started
            output["reconstruction_forward_seconds"] = values.new_tensor(seconds,dtype=torch.float32)
        return output
