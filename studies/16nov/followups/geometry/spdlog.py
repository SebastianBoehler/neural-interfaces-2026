"""Symmetric matrix logarithm with the repeated-eigenvalue Fréchet derivative."""
import torch


class _SPDLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.ndim < 2 or x.shape[-1] != x.shape[-2] or not torch.isfinite(x).all():
            raise ValueError('expected finite square symmetric matrices')
        if not torch.allclose(x, x.transpose(-1, -2), atol=1e-10, rtol=1e-10):
            raise ValueError('matrix must be symmetric')
        w, u = torch.linalg.eigh(x)
        if (w <= 0).any():
            raise ValueError('matrix must be strictly positive definite')
        ctx.save_for_backward(w, u)
        return (u * w.log().unsqueeze(-2)) @ u.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad):
        w, u = ctx.saved_tensors
        left, right = w.unsqueeze(-1), w.unsqueeze(-2)
        diff = left - right
        near = diff.abs() <= 1e-10 * torch.maximum(left, right)
        safe = torch.where(near, torch.ones_like(diff), diff)
        divided = (left.log() - right.log()) / safe
        loewner = torch.where(near, 2.0 / (left + right), divided)
        symgrad = (grad + grad.transpose(-1, -2)) * 0.5
        inner = u.transpose(-1, -2) @ symgrad @ u
        answer = u @ (loewner * inner) @ u.transpose(-1, -2)
        return (answer + answer.transpose(-1, -2)) * 0.5


def spd_log(x):
    return _SPDLog.apply(x)
