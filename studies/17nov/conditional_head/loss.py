"""Softmax-gauge-invariant same-class between-person logit penalty."""
import torch


def penalty(theta, delta):
    w = theta[:, :-1]
    centered = w - w.mean(dim=0, keepdim=True)
    return (delta @ centered.T).square().mean()


def objective(x, y, theta, delta, lam, q):
    w, bias = theta[:, :-1], theta[:, -1]
    logits = x @ w.T + bias
    ce = (torch.logsumexp(logits, 1) - logits.gather(1, y[:, None]).squeeze(1)).mean()
    return ce + w.square().sum() / (2 * .3 * len(y)) + lam * penalty(theta, delta) / q
