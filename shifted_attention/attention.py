"""Masked attention primitives. True means an observation is available."""
from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def masked_rows(x: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid.unsqueeze(-1), x, 0.0)


def accumulation_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def bf16_autocast(x):
    device = x.device.type
    return (x.dtype in (torch.float32, torch.bfloat16) and torch.is_autocast_enabled(device)
            and torch.get_autocast_dtype(device) == torch.bfloat16)


class StateLinear(nn.Linear):
    """Keep the BF16 matrix product, then add bias and carry residual states in FP32."""
    def forward(self, x):
        if bf16_autocast(x):
            y = F.linear(x, self.weight, None).float()
            return y if self.bias is None else y + self.bias.float()
        return super().forward(x)


def _attention(q, k, v, mask_slice: Callable, source_chunk: int, bias_slice=None):
    """Exact attention with a bounded key tile and an explicit empty-row convention.

    Projections have shape [group,head,position,width]. A mask slice may broadcast
    over heads and queries. Detaching the shift used in log-sum-exp is exact, as
    softmax is invariant to any constant shift within the row.
    """
    promote = bf16_autocast(q)
    if promote:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    keys = k.shape[-2]
    if keys <= source_chunk:
        valid = mask_slice(0, keys)
        present = valid.any(dim=-1, keepdim=True)
        safe = valid | ~present
        if bias_slice is None:
            mask = safe
        else:
            mask = bias_slice(0, keys).masked_fill(~safe, -torch.inf).to(q.dtype)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        result = torch.where(present, result, 0.0)
        return result.float() if promote else result

    dtype = accumulation_dtype(q)
    running_max = torch.full(q.shape[:-1] + (1,), -torch.inf, device=q.device, dtype=dtype)
    denominator = torch.zeros_like(running_max)
    numerator = torch.zeros_like(q, dtype=dtype)
    scale = q.shape[-1] ** -0.5
    for start in range(0, keys, source_chunk):
        end = min(keys, start + source_chunk)
        logits = (q @ k[..., start:end, :].transpose(-1, -2)).to(dtype) * scale
        if bias_slice is not None:
            logits = logits + bias_slice(start, end).to(dtype)
        logits = logits.masked_fill(~mask_slice(start, end), -torch.inf)
        new_max = torch.maximum(running_max, logits.detach().amax(dim=-1, keepdim=True))
        origin = torch.where(torch.isfinite(new_max), new_max, 0.0)
        old_scale = (running_max - origin).exp()
        weight = (logits - origin).exp()
        # Keep the value reduction in the accumulator dtype, including under autocast.
        with torch.autocast(device_type=q.device.type, enabled=False):
            numerator = numerator * old_scale + weight @ v[..., start:end, :].to(dtype)
        denominator = denominator * old_scale + weight.sum(dim=-1, keepdim=True)
        running_max = new_max
    safe_denominator = torch.where(denominator > 0, denominator, 1.0)
    result = (numerator / safe_denominator).to(q.dtype)
    return result.float() if promote else result


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = StateLinear(dim, dim)
        self.k_proj = StateLinear(dim, dim)
        self.v_proj = StateLinear(dim, dim)
        self.out_proj = StateLinear(dim, dim)

    def _heads(self, x):
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, query, memory, key_valid, query_valid, source_chunk=None, channel_ids=None, bias=None):
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(memory))
        v = self._heads(self.v_proj(memory))
        # A shared-channel bias is used only by the flat any-variate control.
        source_chunk = source_chunk or memory.shape[1]
        query_chunk = source_chunk if query.shape[1] > source_chunk else query.shape[1]
        results = []
        for start in range(0, q.shape[-2], query_chunk):
            end = min(q.shape[-2], start + query_chunk)
            def relative_bias(a, b):
                same = channel_ids[start:end, None] == channel_ids[None, a:b]
                return torch.where(same[None, None], bias[:, 0, None, None], bias[:, 1, None, None])
            output = _attention(
                q[:, :, start:end], k, v, lambda a, b: key_valid[:, None, None, a:b],
                source_chunk, relative_bias if bias is not None else None)
            results.append(output.transpose(1, 2).flatten(-2))
        context = self.out_proj(torch.cat(results, dim=1))
        return masked_rows(context, query_valid & key_valid.any(dim=-1, keepdim=True))


class ShiftedChannelAttention(nn.Module):
    """Each receiving head reuses its own projection at every candidate shift."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        s, h = config.stream_width, config.heads
        self.q_proj = StateLinear(s, s, bias=False)
        self.k_proj = StateLinear(s, s, bias=False)
        self.v_proj = StateLinear(s, s, bias=False)
        self.out_proj = StateLinear(s, s, bias=False)
        self.gate_proj = None if config.variant == "no_gate" else StateLinear(s, s, bias=False)
        mix = torch.eye(h)[:, 1:].clone()
        if config.variant in {"hard_shifts", "synchronous"}:
            self.register_buffer("R_nonzero", mix if config.variant == "hard_shifts" else mix.zero_())
        else:
            self.R_nonzero = nn.Parameter(mix + 0.001 * torch.randn_like(mix))

    def coefficients(self):
        r = self.R_nonzero.to(accumulation_dtype(self.R_nonzero))
        return torch.cat((1 - r.sum(dim=1, keepdim=True), r), dim=1)

    def forward(self, z, valid):
        # Public layout [B,C,P,s]; the gathered projection cache is [B,P,H,C,e].
        b, c, patches, width = z.shape
        config = self.config
        def project(layer):
            return layer(z).unflatten(-1, (config.heads, -1)).permute(0, 2, 3, 1, 4)
        q, k, v = project(self.q_proj), project(self.k_proj), project(self.v_proj)
        source_ok = valid.transpose(1, 2)
        coeff = self.coefficients()
        result = []
        sensors = torch.arange(c, device=z.device)
        for p0 in range(0, patches, config.patch_chunk_size):
            p1 = min(patches, p0 + config.patch_chunk_size)
            p = torch.arange(p0, p1, device=z.device)
            shape = (b, p1-p0, config.heads, c)
            km = torch.zeros_like(k[:, p0:p1], dtype=accumulation_dtype(k))
            vm = torch.zeros_like(km)
            supported = torch.zeros(shape, dtype=torch.bool, device=z.device)
            complete = torch.ones((b, p1-p0, c), dtype=torch.bool, device=z.device)
            for g, delta in enumerate(config.shifts):
                ids = p + delta
                index = ids.clamp(0, patches-1)
                donor = source_ok.index_select(1, index) & ((ids >= 0) & (ids < patches))[None, :, None]
                weight = coeff[:, g][None, None, :, None, None]
                mask = donor[:, :, None, :, None]
                km = km + torch.where(mask, k.index_select(1, index), 0.0) * weight
                vm = vm + torch.where(mask, v.index_select(1, index), 0.0) * weight
                supported = supported | (donor[:, :, None, :] & (weight[..., 0].detach() != 0))
                if config.common_support:
                    complete = complete & donor
            if config.common_support:
                supported = supported & complete[:, :, None, :]
            supported = supported.flatten(0, 1)
            def mask_slice(start, end):
                other = sensors[:, None] != sensors[None, start:end]
                return supported[:, :, None, start:end] & other[None, None]
            qp = q[:, p0:p1].flatten(0, 1)
            output = _attention(qp, km.flatten(0, 1).to(q.dtype), vm.flatten(0, 1).to(q.dtype),
                                mask_slice, config.source_chunk_size)
            output = output.reshape(b, p1-p0, config.heads, c, -1).permute(0, 3, 1, 2, 4).flatten(-2)
            if self.gate_proj is not None:
                output = output * torch.sigmoid(self.gate_proj(z[:, :, p0:p1]))
            result.append(masked_rows(self.out_proj(output), valid[:, :, p0:p1]))
        return torch.cat(result, dim=2)
