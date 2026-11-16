"""Causal scalar-band SPD bias; output core and PC128 deltas, never a score."""
import numpy as np
import torch


def stream(probability, core_slopes, pc_slopes, mode):
    p = np.asarray(probability, dtype=np.float64)
    core = np.asarray(core_slopes, dtype=np.float64)
    view = np.asarray(pc_slopes, dtype=np.float64)
    if (mode not in ('zero', 'band', 'global') or p.ndim != 2 or p.shape[1] != 3 or
            core.shape != (3, 17) or view.shape != (128, 17) or
            not all(np.isfinite(a).all() for a in (p, core, view)) or
            (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-8)):
        raise ValueError('invalid probabilities, slopes or registered mode')
    out = p.copy(); pc_delta = np.zeros((len(p), 128), dtype=np.float64)
    if mode == 'zero':
        return out, pc_delta, {'updates': 0, 'grad_max': 0., 'bias_norm': 0., 'clipped': 0}
    c = torch.tensor(core); v = torch.tensor(view)
    a = torch.zeros(17 if mode == 'band' else 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a], lr=.005)
    logp = torch.tensor(np.log(np.maximum(p, 1e-12)))
    grad_max = 0.; updates = 0

    def expanded():
        return a if mode == 'band' else a.expand(17)

    for t in range(len(p)):
        if t >= 32 and t % 8 == 0:
            for _ in range(3):
                opt.zero_grad()
                lp = torch.log_softmax(logp[t-32:t] + c @ expanded(), dim=1)
                q = lp.exp(); marginal = q.mean(0)
                loss = -(q * lp).sum(1).mean() + (marginal * marginal.clamp_min(1e-12).log()).sum()
                loss = loss + .01 * a.square().mean()
                loss.backward()
                if a.grad is None or not torch.isfinite(a.grad).all():
                    raise ValueError('non-finite or missing adaptation gradient')
                grad_max = max(grad_max, float(a.grad.abs().max()))
                opt.step()
                with torch.no_grad():
                    a.mul_(min(1., .5 / max(float(a.norm()), 1e-12)))
                updates += 1
        if t >= 32:
            with torch.no_grad():
                bias = expanded()
                out[t] = torch.softmax(logp[t] + c @ bias, dim=0).numpy()
                pc_delta[t] = (v @ bias).numpy()
    norm = float(a.detach().norm())
    if (not np.isfinite(out).all() or not np.isfinite(pc_delta).all() or norm > .500001 or
            (len(p) > 32 and (updates == 0 or grad_max <= 0 or norm <= 0))):
        raise ValueError('non-finite, unbounded or inactive correction')
    return out, pc_delta, {'updates': updates, 'grad_max': grad_max, 'bias_norm': norm,
                           'clipped': int((p < 1e-12).sum())}
