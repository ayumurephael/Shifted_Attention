import numpy as np
import pytest
import torch

from shifted_attention.config import ModelConfig
from shifted_attention.model import ShiftedAttentionModel
from shifted_attention.probes import (InspectHeads,basis_sensitivity,classify,forward_public,pulse_locations,
                                     pulses,support)


def fixture():
    cfg=ModelConfig(patch_width=4,stream_width=12,heads=3,shifts=(0,-1,1),blocks=1,
                    root_seeds=2,point_seeds=2,patch_chunk_size=2,source_chunk_size=2)
    torch.manual_seed(3)
    model=ShiftedAttentionModel(cfg).eval()
    s=dict(values=torch.randn(21,3),observed=torch.ones(21,3,dtype=torch.bool),
           channel_valid=torch.ones(3,dtype=torch.bool),time=torch.arange(21,dtype=torch.float64))
    s["observed"][4:8,2]=False
    return model,s


def test_support_mask_exact_zeros_boundaries_and_visible_donors():
    valid=torch.tensor([[[1,1,1],[0,1,0]]],dtype=torch.bool)
    coeff=torch.tensor([[1.,0.],[1.,1e-6]])
    count,reasons=support(valid,torch.ones((1,2),dtype=torch.bool),(0,1),coeff)
    assert count[0,0,0].tolist()==[0,1]
    assert count[0,0,2].tolist()==[0,0]
    assert reasons[0,0].tolist()==[2,2,3]
    common,_=support(valid,torch.ones((1,2),dtype=torch.bool),(0,1),coeff,True)
    assert common[0,0,0].tolist()==[0,0]


@torch.inference_mode()
def test_observer_leaves_model_exact_and_ignores_private_fields():
    torch.set_num_threads(1)
    model,s=fixture()
    s["private"] = object()
    expected=forward_public(model,s,"cpu")
    seen={}
    with InspectHeads(model,s["channel_valid"][None],seen,64,np.random.default_rng(1)):
        actual=forward_public(model,s,"cpu")
    for key in expected:
        assert torch.equal(actual[key],expected[key])
    counts=[sum(sum(item["support"].values()) for (l,head,r),item in seen.items() if head==h) for h in range(3)]
    assert counts == [17]*3
    assert [r["source_count"] for r in basis_sensitivity(model.blocks[0].channel_attention,"cpu")]==[0,1,1]
    assert sum(item["gate"].count for item in seen.values())==17*12


@torch.inference_mode()
def test_pulses_include_all_phases_adjacent_and_tail():
    torch.set_num_threads(1)
    model,s=fixture()
    s["observed"].fill_(True)
    before=s["values"].clone()
    rows=pulses(model,s,"cpu",lambda y,score,w: .25,11)
    assert set(r["phase"] for r in rows if r["kind"]=="singleton")==set(range(4))
    assert all(r["vus_pr"]==.25 for r in rows)
    assert any(r["kind"]=="last_partial" and r["rows"]%4 for r in rows)
    assert all(r["onset"]//4!=(r["onset"]+r["length"]-1)//4 for r in rows if r["kind"]=="straddle")
    assert torch.equal(s["values"],before)


def test_reachability_lag_and_strict_peaks():
    pytest.importorskip("sster_data")
    d=np.zeros((20,4))
    d[4,0]=1.
    d[9,1]=.75
    d[13,2]=.6
    d[5,3]=8.  # unrelated peak never enters the stratum
    event=[dict(start=4,stop=5)]
    r=classify(d,{0},[{1},{2},set(),set()],1.,3,16,event)
    assert r["reachable_nonroots"]==2 and r["first_symptom_delay"]==5
    assert r["late_onset_after_last_event"] and not r["symptoms_stronger"]
    d[9,1]=1.1
    assert classify(d,{0},[{1},{2},set(),set()],1.,3,16,event)["symptoms_stronger"]
