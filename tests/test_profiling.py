from dataclasses import replace

import pytest
import torch

from shifted_attention.model import ShiftedAttentionModel
from shifted_attention.profiling import forward_flops, profile
from test_model import SMALL
from test_validation import ToyCorpus


@pytest.mark.parametrize("variant",["full","flat","no_gate"])
def test_analytic_flops_match_executed_linear_shapes(variant):
    cfg=replace(SMALL,variant=variant)
    model=ShiftedAttentionModel(cfg).eval()
    linear_flops=[]
    def capture(layer,inputs,result):
        linear_flops.append(2*inputs[0].numel()*layer.out_features)
    handles=[m.register_forward_hook(capture) for m in model.modules() if isinstance(m,torch.nn.Linear)]
    b,t,c=1,11,4
    values=torch.randn(b,t,c)
    with torch.no_grad():
        model(values,torch.ones_like(values,dtype=torch.bool),torch.arange(t).double()[None],
              torch.ones(b,c,dtype=torch.bool))
    for h in handles:
        h.remove()
    p,w,d=3,cfg.patch_width,cfg.final_width
    encoder=4*b*(c*p)**2*d if variant=="flat" else 4*b*(c*p*p+p*c*c)*cfg.stream_width
    readout=4*b*d*(c*cfg.root_seeds*p+c*c+p*w*cfg.point_seeds*c)
    expected=sum(linear_flops)+cfg.blocks*encoder+readout
    assert forward_flops(cfg,b,t,c)["total"]==expected


def test_real_width_profiler_counts_only_measured_windows():
    model=ShiftedAttentionModel(SMALL)
    report=profile(ToyCorpus(),model,[1,4],2,1,torch.device("cpu"))
    assert [r["width"] for r in report]==[1,4]
    assert all(r["measured_windows"]==2 and r["real_rows"]>=4 and r["supervised_forward_gemm_flops"]>0 for r in report)
