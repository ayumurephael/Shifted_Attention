"""Exercise full-key normalization, padding and channel bias independently of optional weight files."""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from shifted_attention.evaluation.author_attention import tiled_forward, install
from shifted_attention.evaluation.baselines import AuthorTimeRCD


class BinaryBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.table=nn.Parameter(torch.tensor([[.4,-.8],[-1.1,.9]]))

    def forward(self,query_id,kv_id):
        same=query_id[:,None,:,None]==kv_id[:,None,None,:]
        return torch.where(same,self.table[1,:,None,None],self.table[0,:,None,None])


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads,self.head_dim,self.embed_dim=2,4,8
        self.q_proj=nn.Linear(8,8,bias=False)
        self.k_proj=nn.Linear(8,8,bias=False)
        self.v_proj=nn.Linear(8,8,bias=False)
        self.out_proj=nn.Linear(8,8,bias=False)
        self.binary_attention_bias=BinaryBias()

    def apply_rope(self,x,freqs):
        a,b=x[...,::2],x[...,1::2]
        c,s=freqs.cos(),freqs.sin()
        return torch.stack((a*c-b*s,a*s+b*c),-1).reshape_as(x)

    def forward(self,query,key,value,freqs,query_id=None,kv_id=None,attn_mask=None):
        b,n,d=query.shape
        q=self.apply_rope(self.q_proj(query),freqs).view(b,n,2,4).transpose(1,2)
        k=self.apply_rope(self.k_proj(key),freqs).view(b,n,2,4).transpose(1,2)
        v=self.v_proj(value).view(b,n,2,4).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(4)
        scores=scores+self.binary_attention_bias(query_id,kv_id)
        scores=scores.masked_fill(~attn_mask[:,None,None,:],float("-inf"))
        return self.out_proj((scores.softmax(-1)@v).transpose(1,2).reshape(b,n,d))


@pytest.mark.parametrize("rows",[1,4,9])
def test_query_tiles_match_dense_full_key_attention(rows):
    torch.manual_seed(20261126)
    m=Attention().eval()
    x=torch.randn(2,11,8)
    f=torch.randn(11,4)
    ids=torch.arange(11)[None].expand(2,-1)//3
    mask=torch.ones((2,11),dtype=torch.bool)
    mask[0,8:]=False
    mask[1,0]=False
    with torch.inference_mode():
        dense=m(x,x,x,f,ids,ids,mask)
        m._eval_native=m.forward
        m._eval_query_rows=rows
        m._eval_dense_limit=0
        m._eval_dense_calls=0
        m._eval_tiles=set()
        out=tiled_forward(m,x,x,x,f,ids,ids,mask)
    torch.testing.assert_close(out,dense,rtol=1e-6,atol=1e-6)
    assert m._eval_tiles=={(11,rows)}


@pytest.mark.skipif("AUTHOR_BASELINE_ASSETS" not in os.environ,reason="requires published author weights and TSB-AD-M archive")
@pytest.mark.parametrize("paper",[True,False])
def test_published_weights_batch_one_and_query_tiles(paper):
    data=Path(os.environ["AUTHOR_BASELINE_ASSETS"])
    device=os.environ.get("AUTHOR_BASELINE_DEVICE","cpu")
    x=pd.read_csv(next((data/"TSB-AD-M").glob("*_SMD_id_1_*")),nrows=237).iloc[:,:-1].to_numpy("float32")
    adapter=AuthorTimeRCD(data,device,paper,96)
    if paper:
        obj=adapter.detector.from_local(adapter.paths["multi"],variant="multi",win_size=96,batch_size=64,device=device)
        obj._ensure_tester(x.shape[1])
        model=obj._tester.model
        baseline=obj.predict(x)
        obj._tester.batch_size=1
        evaluate=obj.predict
    else:
        obj=adapter.tsb_class(win_size=96,input_c=x.shape[1],batch_size=64,device=device,checkpoint=adapter.paths["multi"])
        model=obj.model
        baseline=obj.zero_shot(x)
        obj.batch_size=1
        evaluate=obj.zero_shot
    single=evaluate(x)
    params={k:(v.data_ptr(),v._version) for k,v in model.named_parameters()}
    layers=install(model,37,0)
    tiled=evaluate(x)
    assert params=={k:(v.data_ptr(),v._version) for k,v in model.named_parameters()}
    assert all(layer._eval_tiles=={(228,37)} for layer in layers)
    assert len(tiled)==237
    np.testing.assert_allclose(tiled,single,rtol=1e-5,atol=1e-4)
    np.testing.assert_allclose(tiled,baseline,rtol=1e-5,atol=1e-4)
