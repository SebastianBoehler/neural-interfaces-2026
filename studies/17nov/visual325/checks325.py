"""Real-window stateless batch, padding, gradients, frozen state and timing guards."""
import time
import numpy as np
import torch
from support325 import features, state_sha


def run(renderer, encoder, x, device):
    renderer = renderer.to(device); encoder = encoder.to(device)
    x = x.to(device); before = state_sha(encoder)
    def sync():
        if device == 'cuda': torch.cuda.synchronize()
    def value(signal): return features(encoder, renderer(signal), 1435)
    with torch.no_grad():
        a = value(x[:1]); pair = value(x)
        singles = torch.cat([value(z[None]) for z in x])
        error = float((singles-pair).abs().max())
        if error > (1e-4 if device == 'cpu' else .005): raise ValueError('stateless B1/B4 parity failed')
        value(x[1:2]); repeat = value(x[:1])
        if not torch.equal(a, repeat): raise ValueError('prefix/reset/interleaving failed')
        amp = float((value(2*x[:1])-a).abs().max())
        tail = x[:1].clone(); tail[:,0,-1] += renderer.scale
        last = float((value(tail)-a).abs().max())
        if min(amp,last) < 1e-7: raise ValueError('amplitude/finalsample inactive')
        image = renderer(x)
        neutral = (.5-renderer.rgb_mean)/renderer.rgb_sd
        if not torch.equal(image[:,:,:,-10:], neutral.expand(len(x),3,574,10)):
            raise ValueError('padding became learned/nonneutral')
    xx = x[:1].detach().clone().requires_grad_(True)
    out = value(xx)[0,0]
    derivatives = torch.autograd.grad(out, [xx]+list(renderer.parameters()))
    gradients = {name: float(g.abs().max()) for (name,_),g in zip(renderer.named_parameters(),derivatives[1:])}
    if any(not torch.isfinite(g).all() or float(g.abs().max()) <= 1e-12 for g in derivatives):
        raise ValueError('renderer/input gradient inactive')
    timing = {}
    for batch in (1,4):
        z = x[:1].repeat(batch,1,1)
        with torch.no_grad(): value(z)
        sync(); started = time.monotonic()
        with torch.no_grad():
            for _ in range(3): value(z)
        sync(); timing[f'forward_B{batch}_seconds'] = (time.monotonic()-started)/3
    z = x[:1].repeat(4,1,1)
    sync(); started = time.monotonic(); loss = value(z).square().mean(); loss.backward(); sync()
    timing['renderer_backward_B4_seconds'] = time.monotonic()-started
    renderer.zero_grad(set_to_none=True)
    if state_sha(encoder) != before or any(p.grad is not None for p in encoder.parameters()):
        raise ValueError('frozen encoder changed or acquired gradients')
    return dict(batch_max_abs=error, amplitude=amp, finalsample=last,
                parameter_gradients=gradients, timing=timing, encoder_sha256=before)
