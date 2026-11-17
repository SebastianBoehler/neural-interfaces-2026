"""Independent within-window oscillatory recurrence; no state crosses predictions."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def multiply(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def scan(ar, ai, br, bi):
    """Inclusive affine scan from zero state, using log2(T) composition rounds."""
    stride = 1
    while stride < br.shape[1]:
        cr, ci = multiply(ar[:, stride:], ai[:, stride:], br[:, :-stride], bi[:, :-stride])
        nr, ni = multiply(ar[:, stride:], ai[:, stride:], ar[:, :-stride], ai[:, :-stride])
        br = torch.cat((br[:, :stride], br[:, stride:] + cr), 1)
        bi = torch.cat((bi[:, :stride], bi[:, stride:] + ci), 1)
        ar = torch.cat((ar[:, :stride], nr), 1)
        ai = torch.cat((ai[:, :stride], ni), 1)
        stride *= 2
    return br, bi


def sequential(ar, ai, br, bi):
    """Independent direct recurrence for the engineering oracle, never training."""
    hr, hi = torch.zeros_like(br[:, 0]), torch.zeros_like(bi[:, 0])
    real, imag = [], []
    for t in range(br.shape[1]):
        hr, hi = multiply(ar[:, t], ai[:, t], hr, hi)
        hr, hi = hr + br[:, t], hi + bi[:, t]
        real.append(hr); imag.append(hi)
    return torch.stack(real, 1), torch.stack(imag, 1)


class Oscillator(nn.Module):
    def __init__(self, mode, width=32):
        super().__init__()
        if mode not in ('local', 'static', 'selective'):
            raise ValueError('unregistered transition mode')
        self.mode = mode
        tau = torch.logspace(math.log10(.08), math.log10(2.), width)
        frequency = torch.linspace(2., 40., width)
        self.log_tau = nn.Parameter(tau.log())
        self.frequency = nn.Parameter(torch.logit((frequency - .5) / 54.5))
        self.norm = nn.LayerNorm(width)
        self.drive = nn.Linear(width, 2 * width)
        self.dt = nn.Linear(width, width)
        nn.init.zeros_(self.dt.weight); nn.init.zeros_(self.dt.bias)
        self.mix = nn.Linear(2 * width, width)
        self.gate = nn.Linear(width, width)

    def coefficients(self, x):
        z = self.norm(x)
        # Static and selective start identically; only selective dt receives gradients.
        adjustment = .5 * torch.tanh(self.dt(z))
        if self.mode != 'selective':
            adjustment = adjustment * 0.
        delta = torch.exp(-self.log_tau + adjustment) / 120.
        rho = torch.exp(-delta)
        phase = (2. * math.pi / 120.) * (.5 + 54.5 * torch.sigmoid(self.frequency))
        ar, ai = rho * torch.cos(phase), rho * torch.sin(phase)
        br, bi = self.drive(z).chunk(2, -1)
        gain = torch.sqrt(-torch.expm1(-2. * delta))
        return ar, ai, br * gain, bi * gain

    def forward(self, x):
        ar, ai, br, bi = self.coefficients(x)
        if self.mode == 'local':
            yr, yi = br, bi
        else:
            fr, fi = scan(ar, ai, br, bi)
            rr, ri = scan(ar.flip(1), ai.flip(1), br.flip(1), bi.flip(1))
            yr, yi = .5 * (fr + rr.flip(1)), .5 * (fi + ri.flip(1))
        out = self.mix(torch.cat((yr, yi), -1))
        return x + F.gelu(out) * torch.sigmoid(self.gate(self.norm(x)))


class Decoder(nn.Module):
    def __init__(self, mode, moment_mean, moment_sd):
        super().__init__()
        self.stem = nn.Conv1d(41, 32, 9, padding=4)
        self.layers = nn.ModuleList([Oscillator(mode), Oscillator(mode)])
        self.register_buffer('moment_mean', moment_mean.clone())
        self.register_buffer('moment_sd', moment_sd.clone())
        self.head = nn.Linear(64 + 287, 3)

    def representation(self, x):
        # Keep all 480 samples at120Hz. Current-window CAR does not change the raw branch.
        centered = x - x.mean(1, keepdim=True)
        z = self.stem(centered).transpose(1, 2)
        for layer in self.layers:
            z = layer(z)
        return torch.cat((z.mean(1), z.var(1, unbiased=False).clamp_min(1e-6).log()), 1)

    def moments(self, x):
        frequency = torch.fft.rfftfreq(480, 1. / 120., device=x.device)
        power = torch.fft.rfft(x, dim=-1).abs().square()
        bands = ((1., 4.), (4., 8.), (8., 13.), (13., 20.), (20., 30.), (30., 55.))
        blocks = [x.mean(-1)]
        for low, high in bands:
            mask = (frequency >= low) & (frequency < high)
            blocks.append(power[..., mask].mean(-1).clamp_min(1e-8).log())
        return torch.cat(blocks, 1)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (41, 480):
            raise ValueError('expected unchanged batch of41x480 EEG')
        moment = (self.moments(x) - self.moment_mean) / self.moment_sd
        return self.head(torch.cat((self.representation(x), moment), 1))
