"""Numerical and masking contracts of the production encoder and its readouts."""
import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from shifted_attention.attention import ShiftedChannelAttention, _attention
from shifted_attention.config import ModelConfig
from shifted_attention.model import ShiftedAttentionModel, patch_positions


SMALL = ModelConfig(patch_width=4, stream_width=12, blocks=2, heads=3, shifts=(0,-1,1),
                    root_seeds=2, point_seeds=2, source_chunk_size=2, patch_chunk_size=2)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture
def inputs():
    torch.manual_seed(19)
    x = torch.randn(2, 11, 4)
    obs = torch.rand(2, 11, 4) > .23
    obs[:, 4] = False
    obs[1,:,2] = False
    row = torch.arange(11)[None] < torch.tensor([[11], [8]])
    chan = torch.tensor([[1,1,1,1], [1,1,1,0]], dtype=torch.bool)
    x[~obs] = float("nan")
    clock = 1e12 + torch.rand(2, 11, dtype=torch.float64).cumsum(1)
    return x, obs, clock, chan, row


@pytest.mark.parametrize("variant,total", [("full",14841458), ("synchronous",14841218),
    ("hard_shifts",14841218), ("no_gate",14546546), ("flat",21760226)])
def test_paper_parameter_ledger(variant, total):
    model = ShiftedAttentionModel(ModelConfig(variant=variant))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == total
    head_prefix = ("root_", "point_", "unpatch.")
    assert sum(p.numel() for n,p in model.named_parameters() if n.startswith(head_prefix)) == 7396610


def scalar_shift_reference(layer, z, valid):
    """Loop over target channels, receiving heads, shifts, and donor sensors."""
    b,c,p,s = z.shape
    h,shifts = layer.config.heads, layer.config.shifts
    k = layer.k_proj(z).reshape(b,c,p,h,-1)
    v = layer.v_proj(z).reshape_as(k)
    q = layer.q_proj(z).reshape_as(k)
    r = layer.R_nonzero
    out = torch.zeros_like(q)
    for batch in range(b):
        for target in range(c):
            for t in range(p):
                if not valid[batch,target,t]:
                    continue
                for head in range(h):
                    keys, values = [], []
                    coeff = [1-r[head].sum(), *r[head].unbind()]
                    for src in range(c):
                        if src == target:
                            continue
                        usable = [0 <= t+d < p and bool(valid[batch,src,t+d]) for d in shifts]
                        if layer.config.common_support and not all(usable):
                            continue
                        if not any(ok and bool(w.detach() != 0) for ok,w in zip(usable,coeff)):
                            continue
                        keys.append(sum(w*k[batch,src,t+d,head] for d,w,ok in zip(shifts,coeff,usable) if ok))
                        values.append(sum(w*v[batch,src,t+d,head] for d,w,ok in zip(shifts,coeff,usable) if ok))
                    if keys:
                        scores = torch.stack(keys) @ q[batch,target,t,head] / math.sqrt(s // h)
                        out[batch,target,t,head] = scores.softmax(0) @ torch.stack(values)
    out = out.flatten(-2)
    if layer.gate_proj is not None:
        out = out * layer.gate_proj(z).sigmoid()
    return layer.out_proj(out)


@pytest.mark.parametrize("variant,common", [("synchronous",False),("hard_shifts",False),
    ("full",False),("no_gate",False),("full",True)])
def test_shift_receiving_heads_boundary_mask_and_gate(variant, common):
    torch.manual_seed(103)
    cfg = replace(SMALL, variant=variant, common_support=common)
    module = ShiftedChannelAttention(cfg).double()
    if variant in ("full","no_gate"):
        # Heads 0 and 1 both use +1 and retain distinct projections.
        with torch.no_grad():
            module.R_nonzero.copy_(torch.tensor([[0,1],[0,1],[.8,-.2]]))
    x = torch.randn(2,4,5,12, dtype=torch.float64, requires_grad=True)
    valid = torch.rand(2,4,5) > .21
    valid[1,1:] = False
    ref = scalar_shift_reference(module,x,valid)
    actual = module(x,valid)
    torch.testing.assert_close(actual,ref,rtol=1e-10,atol=1e-12)
    weights = torch.randn_like(actual)
    grads = torch.autograd.grad((actual*weights).sum(), (x,module.v_proj.weight))
    ref_grads = torch.autograd.grad((ref*weights).sum(), (x,module.v_proj.weight))
    for a,b in zip(grads,ref_grads):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)
    assert (actual[1] == 0).all()


def test_exact_source_chunking_outputs_and_gradients():
    torch.manual_seed(19)
    q,k,v = [torch.randn(2,3,n,4,dtype=torch.double,requires_grad=True) for n in (5,9,9)]
    allowed = torch.rand(2,3,5,9) > .55
    allowed[:,0] = False
    allowed[:,1,:,0:4] = False
    def mask(a,b):
        return allowed[...,a:b]
    full = _attention(q,k,v,mask,9)
    chunk = _attention(q,k,v,mask,2)
    torch.testing.assert_close(full,chunk,atol=2e-14,rtol=1e-12)
    top = torch.randn_like(full)
    for a,b in zip(torch.autograd.grad((top*full).sum(),(q,k,v)),
                   torch.autograd.grad((top*chunk).sum(),(q,k,v))):
        torch.testing.assert_close(a,b,atol=2e-14,rtol=1e-12)


def test_clock_translation_padding_and_median():
    clock = torch.tensor([[1,2,5,9,12,15,16,0]],dtype=torch.double)
    rows = torch.tensor([[1,1,1,1,1,1,1,0]],dtype=torch.bool)
    got = patch_positions(clock,rows,4,12)
    # Six positive steps, median 3. Last patch has only three real rows.
    centres = torch.tensor([[(4.25-1)/12, ((12+15+16)/3-1)/12]], dtype=torch.double)
    torch.testing.assert_close(got[...,0],centres.sin())
    torch.testing.assert_close(got[...,1],centres.cos())
    torch.testing.assert_close(got,patch_positions(clock+1e12,rows,4,12))


@pytest.mark.parametrize("variant",["full","flat"])
def test_end_to_end_channel_permutation_and_padding(inputs,variant):
    model = ShiftedAttentionModel(replace(SMALL,variant=variant)).eval()
    x,obs,clock,chan,row = inputs
    order = torch.tensor([2,0,3,1])
    with torch.no_grad():
        out = model(*inputs)
        perm = model(x[:,:,order],obs[:,:,order],clock,chan[:,order],row)
        xp = F.pad(x,(0,2,0,6),value=float("nan"))
        op = F.pad(obs,(0,2,0,6),value=True)
        cp = F.pad(clock,(0,6),value=float("nan"))
        rp = F.pad(row,(0,6))
        ch = F.pad(chan,(0,2))
        padded = model(xp,op,cp,ch,rp)
    effective = obs & chan[:,None] & row[:,:,None]
    assert torch.equal(out["anomaly_valid"], effective.any(-1))
    assert torch.equal(out["root_valid"], effective.any(1))
    for name,key,indices in (("root","root_logits",order), ("anomaly","anomaly_logits",slice(None))):
        assert torch.isfinite(out[key]).all()
        assert (out[key][~out[name+"_valid"]] == 0).all()
        torch.testing.assert_close(out[key][:,indices], perm[key], atol=3e-5,rtol=2e-5)
        torch.testing.assert_close(out[key], padded[key][:,:out[key].shape[1]], atol=3e-5,rtol=2e-5)


def test_stream_isolation_and_singleton(inputs):
    model = ShiftedAttentionModel(SMALL).eval()
    x,obs,clock,chan,row = inputs
    def encode(x,obs,chan):
        token,mask,_ = model.tokenize(x,obs,clock,chan,row)
        return model.encode(token,mask)
    with torch.no_grad():
        normal = encode(x,obs,chan)
        changed = x.clone()
        changed[:,:,1] += 2.0
        perturbed = encode(changed,obs,chan)
        torch.testing.assert_close(normal[:,0,:,:12],perturbed[:,0,:,:12],rtol=0,atol=0)
        assert not torch.allclose(normal[:,0,:,12:],perturbed[:,0,:,12:])
        single = encode(x[:,:,:1],obs[:,:,:1],chan[:,:1])
        assert single[...,12:].count_nonzero() == 0
        empty = model(x,torch.zeros_like(obs),clock,chan,row)
        assert all(t.count_nonzero() == 0 for t in empty.values())


@pytest.mark.parametrize("amp",[False,True])
def test_training_checkpoint_gradient_and_abstention(inputs,amp):
    model = ShiftedAttentionModel(SMALL).train()
    with torch.autocast("cpu",dtype=torch.bfloat16,enabled=amp):
        out = model(*inputs)
        loss = sum(F.softplus(out[k].float()[out[valid]]).mean() for k,valid in
                   (("root_logits","root_valid"),("anomaly_logits","anomaly_valid")))
    loss.backward()
    assert torch.isfinite(loss)
    for n,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert model.blocks[0].channel_attention.R_nonzero.grad.abs().sum() > 0
    assert model.unpatch.weight.grad.abs().sum() > 0
    decay, excluded = model.decay_parameter_groups(.05)
    assert len(decay["params"])+len(excluded["params"]) == len(list(model.parameters()))
    assert {id(p) for p in decay["params"]}.isdisjoint({id(p) for p in excluded["params"]})
    assert id(model.root_pool.seeds) in {id(p) for p in excluded["params"]}


def test_point_recovery_phase_order(inputs):
    model = ShiftedAttentionModel(SMALL).eval()
    x,obs,clock,chan,row = inputs
    token,mask,m = model.tokenize(x,obs,clock,chan,row)
    encoded = model.encode(token,mask)
    with torch.no_grad():
        actual = model._point_chunk(encoded,m)
        reference = []
        for t in range(m.shape[-1]):
            p,offset = divmod(t,SMALL.patch_width)
            d = SMALL.final_width
            sl = slice(d*offset,d*(offset+1))
            recovered = F.linear(encoded[:,:,p],model.unpatch.weight[sl],model.unpatch.bias[sl])
            summary = model.point_pool(recovered,m[:,:,t])
            logit = model.point_score(model.point_compress(summary.flatten(1))).squeeze(-1)
            reference.append(torch.where(m[:,:,t].any(1),logit,0.0))
    torch.testing.assert_close(actual,torch.stack(reference,1),atol=2e-5,rtol=2e-5)
