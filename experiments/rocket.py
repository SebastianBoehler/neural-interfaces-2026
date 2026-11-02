"""MiniRocket-style random-kernel member (overnight 2 Nov), AlignedNet arch ``rocket``.

ROCKET / MiniRocket (Dempster et al. 2020, 2021) classify time series with many fixed
random dilated convolutions, pooled as PPV (fraction of positive outputs) and max, and a
linear classifier. An inductive bias unlike the trained CNN families: no learned filters,
so nothing to overfit to the ten training participants except the linear read-out.
Here: kernels of length 9 at dilations 1-32, each on a random subset of 1-8 channels,
N(0, 1) weights centred to zero sum, biases from quantiles of the outputs on the first
training batch; features standardised by batch norm, then a linear head (AlignedNet
trains only the head and the batch norm). Stateless, so it packages as a jit member.
"""

import torch
from torch import nn

DILATIONS = (1, 2, 4, 8, 16, 32)


class Rocket(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times, n_kernels=1980, **_):
        super().__init__()  # kernels follow the global seed (AlignedNet seeds torch first)
        g = torch.Generator().manual_seed(int(torch.randint(1 << 30, (1,))))
        k = n_kernels // len(DILATIONS)
        self.convs = nn.ModuleList()
        for dl in DILATIONS:
            conv = nn.Conv1d(n_chans, k, 9, dilation=dl, padding=4 * dl)
            w = torch.randn(k, n_chans, 9, generator=g)
            mask = torch.zeros(k, n_chans)
            for i in range(k):
                c = torch.randperm(n_chans, generator=g)[:int(torch.randint(1, 9, (1,), generator=g))]
                mask[i, c] = 1.0
            w = w * mask[:, :, None]
            w = w - w.mean(-1, keepdim=True) * mask[:, :, None]
            conv.weight.data.copy_(w)
            conv.bias.data.zero_()
            conv.weight.requires_grad_(False)
            conv.bias.requires_grad_(False)
            self.convs.append(conv)
        self.q = torch.rand(len(DILATIONS), k, generator=g) * 0.8 + 0.1  # bias quantiles
        self.register_buffer("ready", torch.zeros(()))
        self.bn = nn.BatchNorm1d(2 * k * len(DILATIONS))
        self.final_layer = nn.Linear(2 * k * len(DILATIONS), n_outputs)

    def forward(self, x):
        feats = []
        for i, conv in enumerate(self.convs):
            h = conv(x)
            if self.training and not bool(self.ready):  # MiniRocket: biases from data
                with torch.no_grad():
                    conv.bias.copy_(-torch.quantile(h.transpose(0, 1).flatten(1), self.q[i].to(h), dim=1)
                                    .diagonal())
                h = conv(x)
            feats += [(h > 0).float().mean(-1), h.amax(-1)]
        if self.training:
            self.ready.fill_(1.0)
        return self.final_layer(self.bn(torch.cat(feats, 1)))
