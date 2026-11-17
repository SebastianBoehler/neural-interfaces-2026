"""Fixed source-calibrated amplitude display, explicitly padded for14-pixel patches."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Display(nn.Module):
    def __init__(self, scale, projection):
        super().__init__()
        if not np.isfinite(scale) or scale <= 0 or projection.shape != (16, 41):
            raise ValueError('invalid source display parameters')
        self.register_buffer('scale', torch.tensor(float(scale), dtype=torch.float32))
        self.register_buffer('projection', torch.as_tensor(projection, dtype=torch.float32))
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('rgb_sd', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def signal(self, x, rows):
        if x.ndim != 3 or x.shape[1:] != (41, 480) or not torch.isfinite(x).all():
            raise ValueError('original finite41x480 EEG tensor required')
        if rows == 41:
            return x
        if rows == 16:
            return torch.einsum('rc,bct->brt', self.projection, x)
        raise ValueError('only registered41/16row displays allowed')

    def intensity(self, x, rows):
        return .5 + torch.atan(self.signal(x, rows) / self.scale) / torch.pi

    def forward(self, x, rows):
        v = self.intensity(x, rows)
        image = v.unsqueeze(2).expand(-1, -1, 14, -1).reshape(len(x), 1, rows * 14, 480)
        image = F.pad(image, (0, 10, 0, 0), value=.5).repeat(1, 3, 1, 1)
        if image.shape[2:] != (rows * 14, 490):
            raise ValueError('display cropped or padded incorrectly')
        return (image - self.rgb_mean) / self.rgb_sd


def controls(display, x):
    """Same16projected rows; direct temporal means and unnormalized mean/power."""
    v = display.signal(x, 16) / display.scale
    waveform = v.reshape(len(v), 16, 48, 10).mean(-1).flatten(1)
    f = torch.fft.rfftfreq(480, 1 / 120).to(v.device)
    power = torch.fft.rfft(v).abs().square() / 480 ** 2
    bands = ((.1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 55))
    meanpower = torch.cat([v.mean(-1)] + [power[:, :, (f >= lo) & (f < hi)].mean(-1)
        .clamp_min(1e-12).log() for lo, hi in bands], 1)
    result = torch.cat([waveform, meanpower], 1)
    if result.shape != (len(x), 880):
        raise ValueError('direct control feature dimension differs')
    return result
