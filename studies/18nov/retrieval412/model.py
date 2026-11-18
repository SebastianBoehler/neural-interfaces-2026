"""Retrieve static A6 votes from strictly earlier windows using an EEG metric."""
import torch
from torch import nn
from torch.nn import functional as F


def features(net, x):
    conv = next(net.spatial_conv.children())
    h = conv(net.spectral_filtering(x))
    assert h.ndim == 4 and h.shape[2:] == (1, 480)
    return h.reshape(len(h), -1, 4, 120).square().mean(-1).clamp_min(1e-8).log().flatten(1)


class Metric(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project = nn.Linear(dim, 32, bias=False)

    def forward(self, x):
        return F.normalize(self.project(x), dim=1)


def pair_loss(embedding, labels):
    logits = embedding @ embedding.T / .15
    eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    match = (labels[:, None] == labels[None]) & ~eye
    count = match.sum(1)
    assert (count > 0).any()
    logprob = logits - logits.masked_fill(eye, -torch.inf).logsumexp(1, keepdim=True)
    return -(logprob * match).sum(1)[count > 0].div(count[count > 0]).mean()


def correction(query, history, votes, cyclic=False):
    result = torch.zeros((len(query), 3), dtype=query.dtype)
    for n in range(1, len(query)):
        score = (history[:n] @ query[n]) / .15
        values = votes[:n].roll(1, 0) if cyclic else votes[:n]
        diff = score.softmax(0) @ values - values.mean(0)
        result[n] = .5 * n / (n + 20) * diff
    return result
