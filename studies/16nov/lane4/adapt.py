"""§292D: bounded label-free correction of a frozen TSMNet's SPD features."""

import time

import numpy as np
import torch


def readout(net, matrices, logm, sym_fn):
    z = logm(sym_fn(matrices, lambda w: w.clamp_min(net.eps)))
    i, j = net.iu
    scale = torch.where(i == j, 1.0, 2 ** 0.5).to(z)
    return net.final_layer((z[:, i, j] * scale).float())


def stream(net, matrices, mode, logm, sym_fn):
    """Each call starts a fresh session; updates use prior delivered rows only."""
    assert mode in ("frozen", "diag", "full", "zero")
    d = matrices.shape[-1]
    shape = (d,) if mode == "diag" else (d, d)
    a = torch.zeros(shape, dtype=matrices.dtype, device=matrices.device,
                    requires_grad=True)
    opt = torch.optim.Adam([a], lr=0.005)
    identity = torch.eye(d, dtype=matrices.dtype, device=matrices.device)
    out, updates, grad_max = [], 0, 0.0
    started = time.perf_counter()

    def bias():
        h = torch.diag(a) if mode == "diag" else (a + a.T) * 0.5
        return torch.matrix_exp(h * 0.5)

    for t in range(len(matrices)):
        if mode in ("diag", "full") and t >= 32 and t % 8 == 0:
            past = matrices[t - 32:t]
            for _ in range(3):
                opt.zero_grad()
                b = bias()
                z = readout(net, b @ past @ b.T, logm, sym_fn)
                logp = z.log_softmax(1)
                p = logp.exp()
                marginal = p.mean(0)
                loss = -(p * logp).sum(1).mean()
                loss += (marginal * marginal.clamp_min(1e-8).log()).sum()
                loss += 0.01 * a.square().mean()
                loss.backward()
                assert a.grad is not None and torch.isfinite(a.grad).all()
                grad_max = max(grad_max, float(a.grad.abs().max()))
                opt.step()
                with torch.no_grad():
                    a.mul_(min(1.0, 0.5 / max(float(a.norm()), 1e-12)))
                updates += 1
        with torch.no_grad():
            b = identity if mode in ("frozen", "zero") else bias()
            out.append(readout(net, b @ matrices[t:t + 1] @ b.T,
                               logm, sym_fn).double().softmax(1)[0].cpu())
    if matrices.is_cuda:
        torch.cuda.synchronize()
    result = torch.stack(out).numpy()
    assert np.isfinite(result).all() and np.allclose(result.sum(1), 1)
    norm = float(a.detach().norm())
    assert norm <= 0.500001
    if mode in ("diag", "full") and len(matrices) > 32:
        assert updates > 0 and grad_max > 0 and norm > 0, "inactive SPD update"
    return result, dict(updates=updates, grad_max=grad_max, bias_norm=norm,
                        seconds=time.perf_counter() - started)
