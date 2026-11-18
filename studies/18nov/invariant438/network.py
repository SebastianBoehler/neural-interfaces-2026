"""Encode waveforms once, summarize earlier windows by their learned mean write."""
import torch
from torch import nn
from model import Decoder


class SetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        wave = Decoder()
        self.spatial = wave.spatial
        self.temporal = wave.temporal
        self.write = nn.Sequential(nn.Linear(192, 48), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(241, 96), nn.GELU(), nn.Linear(96, 3))

    def encode(self, x):
        features = self.temporal(self.spatial(x))
        return features.unflatten(-1, (4, -1)).square().mean(-1).clamp_min(1e-8).log().flatten(1)

    def read(self, q, sums, count):
        return self.head(torch.cat((q, sums / count.clamp_min(1), count / (count + 20)), dim=-1))

    def predict_session(self, q, writes):
        prefix = torch.cat((torch.zeros_like(writes[:1]), writes[:-1].cumsum(0)))
        age = torch.arange(len(q), device=q.device, dtype=q.dtype)[:, None]
        return self.read(q, prefix, age)


def subsets(length, gen):
    """Independently sample exactly ceil(age/2) distinct earlier writes for each query."""
    index = torch.arange(length, device='cuda')
    earlier = index[None, :] < index[:, None]
    count = ((index + 1) // 2)[:, None]
    noise = torch.rand((2, length, length), device='cuda', generator=gen)
    rank = noise.masked_fill(~earlier, 2).argsort(-1).argsort(-1)
    mask = rank < count
    assert torch.equal(mask.sum(-1), count.T.expand(2, -1)) and not (mask & ~earlier).any()
    return mask.float(), count
