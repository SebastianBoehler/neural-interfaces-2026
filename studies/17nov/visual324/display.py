"""Fixed intensity and source-calibrated band-limited time/frequency displays."""
import torch
from torch import nn
from torch.nn import functional as F

LAYOUTS = (41, 16)
VIEWS = ('intensity', 'frequency')


class Display(nn.Module):
    def __init__(self, scale, projection, spectral_mean=None, spectral_sd=None):
        super().__init__()
        self.register_buffer('scale', torch.tensor(float(scale), dtype=torch.float32))
        self.register_buffer('projection', torch.as_tensor(projection, dtype=torch.float32))
        self.register_buffer('centers', torch.logspace(0, torch.log10(torch.tensor(55.)).item(), 14))
        frequencies = torch.fft.fftfreq(2048, 1 / 120)
        kernel = torch.exp(-.5 * ((frequencies[None] - self.centers[:, None]) / (self.centers[:, None] / 6)) ** 2)
        kernel[:, (frequencies <= 0) | (frequencies >= 60)] = 0
        kernel /= torch.linalg.vector_norm(kernel, dim=1, keepdim=True)
        self.register_buffer('frequencies', frequencies)
        self.register_buffer('kernel', kernel)
        self.register_buffer('spectral_mean', torch.zeros(2, 14) if spectral_mean is None else torch.as_tensor(spectral_mean).float())
        self.register_buffer('spectral_sd', torch.ones(2, 14) if spectral_sd is None else torch.as_tensor(spectral_sd).float())
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('rgb_sd', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def signal(self, x, rows):
        if x.ndim != 3 or x.shape[1:] != (41, 480) or not torch.isfinite(x).all():
            raise ValueError('finite original41x480 tensor required')
        if rows == 41: return x / self.scale
        if rows == 16: return torch.einsum('rc,bct->brt', self.projection, x) / self.scale
        raise ValueError('unknown spatial representation')

    def logpower(self, x, rows):
        value = F.pad(self.signal(x, rows), (784, 784))
        z = torch.fft.fft(value)
        analytic = 2 * torch.fft.ifft(z[:, :, None, :] * self.kernel[None, None, :, :])
        return analytic[:, :, :, 784:1264].abs().square().clamp_min(1e-12).log()

    def plane(self, x, rows, view):
        if view == 'intensity':
            v = .5 + torch.atan(self.signal(x, rows)) / torch.pi
            return v.unsqueeze(2).expand(-1, -1, 14, -1)
        if view == 'frequency':
            i = LAYOUTS.index(rows)
            z = (self.logpower(x, rows) - self.spectral_mean[i][None, None, :, None]) / self.spectral_sd[i][None, None, :, None]
            return .5 + torch.atan(z / 3) / torch.pi
        raise ValueError('unknown display view')

    def forward(self, x, rows, view):
        plane = self.plane(x, rows, view).reshape(len(x), 1, rows * 14, 480)
        image = F.pad(plane, (0, 10, 0, 0), value=.5).repeat(1, 3, 1, 1)
        if image.shape[2:] != (rows * 14, 490):
            raise ValueError('image lost samples or frequency rows')
        return (image - self.rgb_mean) / self.rgb_sd

    def raw_meanpower(self, x):
        v = x / self.scale
        f = torch.fft.rfftfreq(480, 1 / 120).to(x.device)
        p = torch.fft.rfft(v).abs().square() / 480 ** 2
        bands = ((.1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 55))
        return torch.cat([v.mean(-1)] + [p[:, :, (f >= lo) & (f < hi)].mean(-1).clamp_min(1e-12).log() for lo, hi in bands], 1)

    def direct(self, x, rows, view):
        if view == 'intensity':
            base = self.signal(x, rows).reshape(len(x), rows, 48, 10).mean(-1).flatten(1)
        elif view == 'frequency':
            base = self.logpower(x, rows).reshape(len(x), rows, 14, 8, 60).mean(-1).flatten(1)
        else: raise ValueError('unknown direct control')
        return torch.cat([base, self.raw_meanpower(x)], 1)
