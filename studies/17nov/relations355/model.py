"""Task-learned full-montage relations; no cross-window state or sensor projection."""
import torch
from torch import nn
from torch.nn import functional as F

ARMS = {'lag': ('instant', 'lag6'), 'envelope': ('offset', 'aligned')}


class Relations(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in sum((list(v) for v in ARMS.values()), []):
            raise ValueError(arm)
        self.arm = arm
        # One shared temporal bank is applied separately to every native sensor.
        self.filters = nn.Conv1d(1, 4, 65, padding=32, bias=False)
        width = 4 * 41 * 41 + 82
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ELU(), nn.Linear(64, 3))

    def representation(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (41, 480):
            raise ValueError('Expected unchanged official 41x480 tensor')
        x = (x - x.mean(1, keepdim=True)) / 100.
        skip = torch.cat((x.mean(-1), (x.square().mean(-1) + 1e-6).log()), 1)
        z = self.filters(x.reshape(-1, 1, 480)).reshape(-1, 41, 4, 480).transpose(1, 2)
        if self.arm in ('aligned', 'offset'):
            z = F.avg_pool1d(z.flatten(0, 2).unsqueeze(1).square(), 25, 1, 12)
            z = z.reshape(-1, 4, 41, 480)
            if self.arm == 'offset':
                # Fixed row-independent offsets destroy simultaneous envelope alignment.
                index = (torch.arange(480, device=x.device)[None, None, :] +
                         37 * torch.arange(41, device=x.device)[None, :, None] +
                         71 * torch.arange(4, device=x.device)[:, None, None]) % 480
                z = z.gather(-1, index[None].expand(len(x), -1, -1, -1))
            z = z - z.mean(-1, keepdim=True)
            z = z / (z.square().mean(-1, keepdim=True) + 1e-5).sqrt()
            rel = z @ z.roll(-1, 1).transpose(-1, -2) / 480
        else:
            z = z - z.mean(-1, keepdim=True)
            z = z / (z.square().mean(-1, keepdim=True) + 1e-5).sqrt()
            # Equal 474-sample exposure in both arms, only the partner lag differs.
            right = z[..., 6:] if self.arm == 'lag6' else z[..., :474]
            rel = z[..., :474] @ right.transpose(-1, -2) / 474
        return torch.cat((rel.flatten(1), skip), 1)

    def forward(self, x):
        return self.head(self.norm(self.representation(x)))
