"""Bound the quadratic temporaries of published TimeRCD inference using exact query-row tiles."""
import math
from types import MethodType

import torch


def tiled_forward(layer, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
    batch, length, dim = query.shape
    heads, depth = layer.num_heads,layer.head_dim
    rows = min(length,layer._eval_query_rows)
    # Cap one [B,H,rows,L] temporary at 512 MiB even on the 248-variate benchmark.
    rows = min(rows,max(1, (512*1024**2)//(batch*heads*length*query.element_size())))
    if length <= layer._eval_dense_limit or query_id is None or kv_id is None:
        layer._eval_dense_calls += 1
        return layer._eval_native(query,key,value,freqs,query_id,kv_id,attn_mask)
    layer._eval_tiles.add((length,rows))
    def heads_view(x):
        return x.reshape(batch,length,heads,depth).transpose(1,2)
    q = heads_view(layer.apply_rope(layer.q_proj(query),freqs))
    k = heads_view(layer.apply_rope(layer.k_proj(key),freqs)).transpose(-2,-1)
    v = heads_view(layer.v_proj(value))
    output = torch.empty_like(q)
    invalid = None if attn_mask is None else ~attn_mask[:,None,None,:]
    for a in range(0,length,rows):
        b = min(a+rows,length)
        scores = torch.matmul(q[:,:,a:b],k)
        scores.div_(math.sqrt(depth))
        scores.add_(layer.binary_attention_bias(query_id[:,a:b],kv_id))
        if invalid is not None:
            scores.masked_fill_(invalid,float("-inf"))
        weights = torch.softmax(scores,dim=-1)
        del scores
        output[:,:,a:b] = torch.matmul(weights,v)
        del weights
    return layer.out_proj(output.transpose(1,2).contiguous().reshape(batch,length,dim))


def install(model,query_rows=128,dense_limit=2048):
    if query_rows<1 or dense_limit<0:
        raise ValueError("query_rows must be positive and dense_limit nonnegative")
    layers = [block.self_attn for block in model.ts_encoder.transformer_encoder.layers]
    for layer in layers:
        if layer.__class__.__name__!="MultiheadAttentionWithRoPE" or layer.binary_attention_bias is None:
            raise ValueError("Unexpected multivariate author attention module")
        if not hasattr(layer,"_eval_native"):
            layer._eval_native = layer.forward
            layer.forward = MethodType(tiled_forward,layer)
        layer._eval_query_rows = query_rows
        layer._eval_dense_limit = dense_limit
        layer._eval_tiles = set()
        layer._eval_dense_calls = 0
    return layers
