"""Run the task-private motor path without invoking native SessionBN."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from external import EVENT, PREFIX, TOL
from source_targets import normalizer


PRIVATE_PARAMS=3461


class MotorHead(nn.Module):
    def __init__(self,seed,device):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(32000+seed)
            self.norm=nn.BatchNorm2d(288,track_running_stats=False)
            self.task=nn.Linear(576,5)
        self.to(device)


def motor(model,head,x,detached):
    body=model.body
    core=body.spatial_conv
    if len(core)!=3 or type(core[2]) is not torch.nn.SiLU:
        raise ValueError("unexpected native trainable spatial stack")
    value=core[0](body.spectral_filtering(x)[...,:PREFIX])
    if value.shape[1:]!=(288,1,PREFIX) or head.norm.track_running_stats:
        raise ValueError("external geometry or isolated stateless normalization differs")
    pre=core[2](head.norm(value.detach() if detached else value))
    return head.task(body.temporal_layer(pre.reshape(len(pre),288,2,120)).flatten(1))


@torch.no_grad()
def check_tail(model,head,x):
    bank=model.body.spectral_filtering
    taps=[v.numel() for v in bank.fir_list]
    radius=max(taps)//2
    if bank.method_iir or bank.forward_filter or len(taps)!=9 or radius+PREFIX>EVENT or any(n%2==0 for n in taps):
        raise ValueError("external labeled window cannot exclude the post-event tail")
    original=motor(model,head,x,True)
    changed=x.clone()
    changed[...,EVENT:]=x[...,:480-EVENT].flip(-1)
    drift=float((motor(model,head,changed,True)-original).abs().max())
    if not 0<=drift<=TOL:
        raise ValueError(f"external post-event waveform leaked into the auxiliary head: {drift}")
    return dict(filter_radius=radius, tail_drift=drift)


def inspect_gradient(model,head,loss,detached):
    layer=normalizer(model)
    stem=model.body.spatial_conv[0].parametrizations.weight.original
    spatial,affine,bias,aux,private=torch.autograd.grad(
        loss,(stem,layer.weight,layer.bias,head.task.weight,head.norm.weight),retain_graph=True,allow_unused=True)
    if affine is not None or bias is not None or (spatial is None)!=detached:
        raise ValueError("detached or joint gradient accidentally shared the main normalization")
    values=dict(external_body_gradient=0. if spatial is None else float(spatial.abs().max()),
                external_head_gradient=float(aux.abs().max()),external_norm_gradient=float(private.abs().max()))
    if (not np.isfinite(list(values.values())).all() or min(values["external_head_gradient"],values["external_norm_gradient"])<=1e-8
            or not detached and values["external_body_gradient"]<=1e-8):
        raise ValueError("actual private motor path or shared gradient was inactive")
    return values


@torch.no_grad()
def diagnostic(model,head,fx,fy,tape):
    scores=torch.cat([motor(model,head,fx[r],True) for r in tape[:6]])
    truth=fy[tape[:6]].flatten()
    if not scores.isfinite().all():
        raise ValueError("invalid source external diagnostic")
    return dict(external_final_loss=float(F.cross_entropy(scores,truth)),
                external_final_accuracy=float((scores.argmax(1)==truth).float().mean()))
