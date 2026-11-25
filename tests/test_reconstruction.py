import pytest
import torch

from shifted_attention.config import ModelConfig
from shifted_attention.model import ShiftedAttentionModel
from shifted_attention.train import reconstruction_coordinates
from shifted_attention.evaluation.inference import Predictor


@pytest.mark.parametrize("variant",["full","flat"])
def test_second_forward_keeps_clean_heads_and_masked_values_private(variant,tmp_path):
    torch.manual_seed(27)
    cfg=ModelConfig(variant=variant,stream_width=12,blocks=1,aux_reconstruction=True)
    m=ShiftedAttentionModel(cfg)
    v=torch.randn(1,16,3)
    obs=torch.ones_like(v,dtype=torch.bool)
    obs[0,2,0]=False
    times=torch.arange(16,dtype=torch.float64)[None]
    channels=torch.ones(1,3,dtype=torch.bool)
    rows=torch.ones(1,16,dtype=torch.bool)
    mask=torch.zeros((1,3,2),dtype=torch.bool)
    mask[0,0,0]=True
    inputs=(v,obs,times,channels,rows)
    masked=m(*inputs,reconstruction_mask=mask)
    plain=m(*inputs)
    torch.testing.assert_close(masked["root_logits"],plain["root_logits"])
    torch.testing.assert_close(masked["anomaly_logits"],plain["anomaly_logits"])
    masked["reconstruction_sse"].backward()
    assert m.mask_embedding.grad is not None and m.mask_embedding.grad.abs().sum()>0
    assert m.reconstructor[0].weight.grad.abs().sum()>0
    perturbed=v.clone()
    perturbed[~obs]=10000
    torch.testing.assert_close(masked["reconstruction_sse"],m(perturbed,*inputs[1:],
                               reconstruction_mask=mask)["reconstruction_sse"])
    m.zero_grad(set_to_none=True)
    empty=m(*inputs,reconstruction_mask=torch.zeros_like(mask))["reconstruction_sse"]
    empty.backward()
    assert empty.item()==0 and m.reconstructor[-1].weight.grad is not None
    path=tmp_path/"aux.pt"
    torch.save({"model_config":cfg.__dict__,"model":m.state_dict()},path)
    inference=Predictor(path,"cpu").model
    assert not hasattr(inference,"mask_embedding")
    torch.testing.assert_close(inference(*inputs)["anomaly_logits"],plain["anomaly_logits"])


def test_mask_rng_and_global_denominator_ignore_padding():
    obs=torch.ones(1,37,11,dtype=torch.bool)
    obs[0,:,2]=False
    batch=dict(observed=obs,row_valid=torch.ones(1,37,dtype=torch.bool),
               channel_valid=torch.ones(1,11,dtype=torch.bool),index=torch.tensor([291]))
    small,n=reconstruction_coordinates(batch,8,17,0)
    batch["observed"]=torch.nn.functional.pad(obs,(0,5,0,29))
    batch["row_valid"]=torch.nn.functional.pad(batch["row_valid"],(0,29))
    batch["channel_valid"]=torch.nn.functional.pad(batch["channel_valid"],(0,5))
    padded,k=reconstruction_coordinates(batch,8,17,0)
    assert n==k and n>0 and torch.equal(padded[:,:11,:5],small)
    assert not padded[:,11:].any() and not padded[:,:,5:].any() and not small[:,2].any()
