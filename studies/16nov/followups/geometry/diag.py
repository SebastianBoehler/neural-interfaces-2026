"""Prefix-only learned diagonal SPD congruence and its rebuilt-core posterior delta."""
import numpy as np
import torch

from spdlog import spd_log


def logits_delta(matrices, original_log, head, bias, scalar=False):
    """Linear-head change after congruence, with original matrix logs held fixed."""
    d = matrices.shape[-1]
    i, j = torch.triu_indices(d, d, device=matrices.device)
    if scalar:
        diagonal = i == j
        return (head[:, :, diagonal].sum(dim=(1, 2)) * bias[0]).expand(len(matrices), 3)
    gain = torch.exp(bias * .5)
    changed = gain.view(1, 1, d, 1) * matrices * gain.view(1, 1, 1, d)
    diff = spd_log(changed) - original_log
    scale = torch.where(i == j, 1., 2 ** .5).to(diff)
    vector = diff[..., i, j] * scale
    return torch.einsum('nbk,cbk->nc', vector, head)


def stream(matrices, base_probability, head, mode, cap):
    """Return additive reconstructed-posterior deltas; no archived data are altered."""
    p = np.asarray(base_probability, dtype=np.float64)
    if (not isinstance(matrices, torch.Tensor) or matrices.ndim != 4 or
            matrices.shape[0] != len(p) or matrices.shape[-1] != matrices.shape[-2] or
            p.shape != (len(matrices), 3) or not np.isfinite(p).all() or (p <= 0).any() or
            not np.allclose(p.sum(1), 1, atol=1e-8) or mode not in ('zero', 'scalar', 'diag') or
            cap not in (0., .25, .5, 1.)):
        raise ValueError('invalid session matrices, posterior, mode or cap')
    d, blocks = matrices.shape[-1], matrices.shape[1]
    expected = (3, blocks, d * (d + 1) // 2)
    if tuple(head.shape) != expected or not torch.isfinite(head).all() or not torch.isfinite(matrices).all():
        raise ValueError('invalid frozen core weights or matrices')
    delta = np.zeros_like(p)
    if mode == 'zero' and cap == 0:
        return delta, {'updates': 0, 'grad_max': 0., 'bias_norm': 0.}
    if mode == 'zero' or cap == 0 or (mode == 'scalar' and cap != .5):
        raise ValueError('unregistered mode/cap')
    scalar = mode == 'scalar'
    head = head.to(device=matrices.device, dtype=torch.float64)
    original = spd_log(matrices).detach() if not scalar else None
    logp = torch.tensor(np.log(p), dtype=torch.float64, device=matrices.device)
    a = torch.zeros(1 if scalar else d, dtype=torch.float64, device=matrices.device, requires_grad=True)
    optimizer = torch.optim.Adam([a], lr=.005)
    grad_max = 0.; updates = 0
    for t in range(len(p)):
        if t >= 32 and t % 8 == 0:
            for _ in range(3):
                optimizer.zero_grad()
                base = None if scalar else original[t-32:t]
                shift = logits_delta(matrices[t-32:t], base, head, a, scalar)
                lp = torch.log_softmax(logp[t-32:t] + shift, dim=1)
                q = lp.exp(); marginal = q.mean(0)
                loss = -(q * lp).sum(1).mean() + (marginal * marginal.clamp_min(1e-12).log()).sum()
                loss = loss + .01 * a.square().mean()
                loss.backward()
                if a.grad is None or not torch.isfinite(a.grad).all():
                    raise ValueError('missing or non-finite geometry gradient')
                grad_max = max(grad_max, float(a.grad.abs().max()))
                optimizer.step()
                with torch.no_grad():
                    a.mul_(min(1., cap / max(float(a.norm()), 1e-12)))
                updates += 1
        if t >= 32:
            with torch.no_grad():
                base = None if scalar else original[t:t+1]
                shift = logits_delta(matrices[t:t+1], base, head, a, scalar)[0]
                delta[t] = torch.softmax(logp[t] + shift, 0).cpu().numpy() - p[t]
    norm = float(a.detach().norm())
    if (not np.isfinite(delta).all() or norm > cap + 1e-6 or
            (len(p) > 32 and (updates <= 0 or grad_max <= 0 or norm <= 0))):
        raise ValueError('inactive, unbounded or non-finite geometry correction')
    return delta, {'updates': updates, 'grad_max': grad_max, 'bias_norm': norm}
