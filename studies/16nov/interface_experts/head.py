"""Weighted joint softmax head with symmetric signed expert deviations."""
import numpy as np
import torch


def objective(theta, x, y, gate, weights, lam, c=.3):
    n, d = x.shape
    if theta.shape != (2, 3, d + 1) or gate.shape != (n,) or weights.shape != (n,):
        raise ValueError('joint objective shape mismatch')
    shared = x @ theta[0, :, :d].T + theta[0, :, d]
    deviation = x @ theta[1, :, :d].T + theta[1, :, d]
    logits = shared + gate[:, None] * deviation
    loss = torch.logsumexp(logits, 1) - logits.gather(1, y[:, None]).squeeze(1)
    penalty = (theta[0, :, :d].square().sum() + lam * theta[1, :, :d].square().sum()) / (2 * c * n)
    return (weights * loss).mean() + penalty


def fit(x, labels, signed_gate, weights, lam, *, shared=False, max_iter=600):
    device = torch.device('cpu')
    X = torch.as_tensor(x, dtype=torch.float64, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    gate = torch.as_tensor(signed_gate, dtype=torch.float64, device=device)
    w = torch.as_tensor(weights, dtype=torch.float64, device=device)
    if (not torch.isfinite(X).all() or not torch.isfinite(gate).all() or
            not torch.isfinite(w).all() or abs(float(w.sum()) - len(X)) > 1e-7 or
            torch.any((y < 0) | (y > 2))):
        raise ValueError('invalid head fit inputs')
    if shared: gate = torch.zeros_like(gate)
    theta = torch.zeros((2, 3, X.shape[1] + 1), dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1., max_iter=max_iter, tolerance_grad=1e-8,
                            tolerance_change=1e-13, history_size=15, line_search_fn='strong_wolfe')
    evaluations = 0
    def closure():
        nonlocal evaluations
        opt.zero_grad(); loss = objective(theta, X, y, gate, w, lam)
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        loss.backward(); evaluations += 1
        if not torch.isfinite(theta.grad).all(): raise ValueError('nonfinite gradient')
        return loss
    opt.step(closure)
    value = closure(); max_gradient = float(theta.grad.abs().max().detach().cpu())
    result = theta.detach().cpu().numpy().copy()
    if max_gradient > 1e-5 or not np.isfinite(result).all():
        raise ValueError(f'joint head failed convergence: {max_gradient=}')
    if shared and np.any(result[1]): raise ValueError('shared head gained a deviation')
    if not shared and np.linalg.norm(result[1]) <= 1e-10:
        raise ValueError('expert deviation inactive')
    return result, dict(loss=float(value.detach().cpu()), max_gradient=max_gradient,
                        evaluations=evaluations, deviation_norm=float(np.linalg.norm(result[1])))


def predict(theta, z, signed_gate):
    n, d = z.shape
    if theta.shape != (2, 3, d + 1) or np.shape(signed_gate) != (n,):
        raise ValueError('head prediction shape mismatch')
    a = z @ theta[0, :, :d].T + theta[0, :, d]
    b = z @ theta[1, :, :d].T + theta[1, :, d]
    logits = a + np.asarray(signed_gate)[:, None] * b
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    if not np.isfinite(probs).all(): raise ValueError('invalid head output')
    return probs
