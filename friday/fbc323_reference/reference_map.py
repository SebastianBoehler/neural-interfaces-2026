"""Two fixed, energy-matched rank-one interventions on the real waveform axis."""
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
KINDS = ('hard', 'car', 'sham')
MODES = ('ordinary', 'bounded')
ARMS = tuple(f'{kind}_{mode}' for kind in KINDS for mode in MODES)
NATIVE_PARAMETERS = 347903
CHANNELS = 41


def direction(x):
    ids = torch.arange(CHANNELS, device=x.device)
    return ((1 - 2 * ids.remainder(2)).to(x.dtype) / CHANNELS**.5)[:, None]


def reference(x, kind):
    if x.shape[-2:] != (CHANNELS, 480) or x.dtype != torch.float32:
        raise ValueError('unexpected native waveform shape or type')
    car = x - x.mean(dim=-2, keepdim=True)
    if kind == 'car':
        return car
    if kind != 'sham':
        raise ValueError(f'invalid active reference map {kind}')
    v = direction(x)
    artificial = x - v * (v*x).sum(-2, keepdim=True)
    norm_car = car.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    norm_artificial = artificial.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    if not (norm_artificial > 0).all():
        raise ValueError("artificial reference has no remaining signal")
    return artificial * (norm_car / norm_artificial)


def attach(model, kind, calls):
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != NATIVE_PARAMETERS or model.skip:
        raise ValueError('unexpected F16 native body')
    if kind == 'hard':
        return None

    def pre_forward(_, args):
        if len(args) != 1:
            raise ValueError('unexpected native waveform body call')
        calls[0] += 1
        return (reference(args[0], kind),)

    return model.body.register_forward_pre_hook(pre_forward)


@torch.no_grad()
def probe(x, kind):
    if kind == 'hard':
        return dict(delta=0., orthogonal=0., norm_error=0., common_offset=0.)
    y = reference(x, kind)
    car = reference(x, 'car')
    x_norm = x.square().sum(dim=(-2,-1)).sqrt()
    car_norm, y_norm = (z.square().sum(dim=(-2,-1)).sqrt() for z in (car, y))
    if not torch.isfinite(y).all() or not (y_norm > 0).all() or not (x_norm > 0).all():
        raise ValueError('active reference had an undefined or nonfinite norm')
    normal = torch.ones_like(direction(x))/CHANNELS**.5 if kind == 'car' else direction(x)
    energy_error = float(((y_norm-car_norm).abs()/car_norm).max())
    orthogonal = float(((y*normal).sum(-2).abs().amax(-1)/y_norm).max())
    delta = float((x-y).abs().max())
    offset = .23*torch.linspace(-1.,1.,x.shape[-1], device=x.device, dtype=x.dtype)
    common = float((reference(x+offset, kind)-y).abs().max())
    if (min(delta, float(car_norm.min())) <= 1e-5 or energy_error > 2e-6 or orthogonal > 2e-6 or
            (kind == 'car' and common > 5e-6) or (kind == 'sham' and common < .02)):
        raise ValueError(f'{kind} waveform probe failed execution or geometry')
    return dict(delta=delta, orthogonal=orthogonal, norm_error=energy_error, common_offset=common)
