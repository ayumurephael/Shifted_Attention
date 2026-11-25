import numpy as np
import pytest
import torch

from shifted_attention.attention import StateLinear, _attention
from shifted_attention.config import ModelConfig
from shifted_attention.model import ShiftedAttentionModel
from shifted_attention.physical_lags import accumulate_lags, head_lags, physical_lags
from shifted_attention.precision import check_precision


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_state_linear_preserves_full_precision_and_checkpoint_layout(dtype):
    torch.manual_seed(318)
    projection=StateLinear(8,5).to(dtype)
    ordinary=torch.nn.Linear(8,5).to(dtype)
    ordinary.load_state_dict(projection.state_dict(),strict=True)
    x=torch.randn(4,8,dtype=dtype)
    assert torch.equal(projection(x),ordinary(x))
    projection(x).sum().backward()
    ordinary(x).sum().backward()
    assert torch.equal(projection.weight.grad,ordinary.weight.grad)
    assert torch.equal(projection.bias.grad,ordinary.bias.grad)


def test_bf16_projections_residuals_and_attention_state():
    torch.manual_seed(647)
    projection=StateLinear(8,8)
    x=torch.randn(2,4,8)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        product=torch.nn.functional.linear(x,projection.weight,None)
        projected=projection(x)
        assert product.dtype==torch.bfloat16 and projected.dtype==torch.float32
        assert torch.equal(projected,product.float()+projection.bias)
        q=projected[:,None]
        mask=torch.tensor([[[[True,False,True,True]]],[[[False,False,False,False]]]])
        dense=_attention(q,q,q,lambda a,b:mask[...,a:b],4)
        chunk=_attention(q,q,q,lambda a,b:mask[...,a:b],2)
        assert dense.dtype==chunk.dtype==torch.float32
        assert dense[1].count_nonzero()==chunk[1].count_nonzero()==0
        torch.testing.assert_close(dense[:1],chunk[:1],atol=.007,rtol=.005)
        (dense.sum()+chunk.sum()).backward()
    assert torch.isfinite(projection.weight.grad).all()


@pytest.mark.parametrize("common",[False,True])
def test_observed_physical_lag_matches_independent_edge_enumeration(common):
    clock=np.array([[-.5,.5,2,4,8,16,20],[1e12,1e12+1,1e12+2,1e12+4,np.nan,np.nan,np.nan]])
    row=np.isfinite(clock)
    valid=np.array([[[1,0,1,0],[0,1,0,1],[0,1,1,0]],
                    [[1,1,0,0],[0,1,0,0],[0,0,0,0]]],dtype=bool)
    shifts=(0,-1,1)
    ledgers=physical_lags(clock,row,valid,2,shifts,common)
    assert ledgers[0]["centers_from_origin"]==[.5,3.5,12.5,20.5]
    assert ledgers[1]["centers_from_origin"]==[.5,3.,None,None]
    totals={}
    for bi,ledger in enumerate(ledgers):
        accumulate_lags(ledger,totals)
        packed=np.frombuffer(bytes.fromhex(ledger["token_valid_little_endian_hex"]),dtype="u1")
        mask=np.unpackbits(packed,bitorder="little")[:valid[bi].size].reshape(ledger["token_valid_shape"])
        np.testing.assert_array_equal(mask,valid[bi])
        for stats,d in zip(ledger["offsets"],shifts):
            edges=[]
            for p in range(4):
                count=sum(1 for i in range(3) for j in range(3) if
                          i!=j and valid[bi,i,p] and 0<=p+d<4 and valid[bi,j,p+d] and
                          (not common or all(0<=p+k<4 and valid[bi,j,p+k] for k in shifts)))
                if count:
                    centers=ledger["centers_from_origin"]
                    edges.append((p,centers[p+d]-centers[p],count))
            assert list(zip(stats["target_patch"],stats["donor_minus_target"],stats["ordered_channel_pairs"]))==edges
    heads=head_lags(shifts,[0,-1,1e-12],totals)
    assert heads[0]["ordered_channel_pairs"]==0 and heads[0]["mean"] is None
    if not common:
        assert heads[1]["maximum"]<0 and heads[2]["minimum"]>0
        assert heads[1]["ordered_channel_pairs"]==heads[2]["ordered_channel_pairs"]
    else:
        assert all(h["ordered_channel_pairs"]==0 for h in heads)


def test_precision_compares_paired_inputs_and_reports_thresholds():
    torch.manual_seed(281)
    torch.set_num_threads(1)
    cfg=ModelConfig(patch_width=4,stream_width=12,heads=3,shifts=(0,-1,1),blocks=2)
    model=ShiftedAttentionModel(cfg).eval()
    report=check_precision(model,"cpu",seed=445,draws=2)
    assert report["mask_mismatches"]=={"fp32_vs_fp64":0,"bf16_vs_fp32":0}
    assert [x["operator_layer"] for x in report["draws"]]==[0,1]
    for kind,results in report["errors"].items():
        for point in results.values():
            assert 0<=point["p50"]<=point["p95"]<=point["p99"]<=point["maximum"]
            assert point["target"]==(1e-5 if kind=="fp32_vs_fp64" else .02)
            assert point["maximum_within_target"]==(point["count_above_target"]==0)
            assert point["count"]>0
    assert all(x["maximum_within_target"] for x in report["errors"]["fp32_vs_fp64"].values())
