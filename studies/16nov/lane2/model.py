"""Lane 2: fixed per-window features and a population-trained neural decoder."""
import numpy as np
import torch
from torch import nn

BANDS = ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))


def features(x):
    """No session statistics: only the supplied four-second window is used."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (41, 480) or not np.isfinite(x).all():
        raise ValueError(f"expected finite (n, 41, 480) windows, got {x.shape}")
    mean = x.mean(axis=2, dtype=np.float64)
    f = np.fft.rfftfreq(480, d=1 / 120)
    power = np.abs(np.fft.rfft(x - mean[..., None], axis=2)) ** 2 / 480**2
    bands = [np.log(np.maximum(power[:, :, (f >= lo) & (f < hi)].mean(axis=2), 1e-20))
             for lo, hi in BANDS]
    return np.concatenate([mean, *bands], axis=1).astype(np.float32)


class Head(nn.Module):
    def __init__(self, n_in=246):
        super().__init__()
        self.hidden = nn.Linear(n_in, 64)
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        return self.out(torch.nn.functional.gelu(self.hidden(x)))


def train_population(x, y, person, seed, device, epochs=60):
    """Class- and person-balanced weighted source loss; never sees test rows."""
    x, y, person = np.asarray(x), np.asarray(y), np.asarray(person).astype(str)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(person):
        raise ValueError("training shapes differ")
    if set(np.unique(y)) != {0, 1, 2} or len(set(person)) != 9 or not np.isfinite(x).all():
        raise ValueError("expected nine training people, three classes, finite features")
    mu = x.mean(0); sd = np.maximum(x.std(0), 1e-4)
    xx = torch.as_tensor(np.clip((x - mu) / sd, -8, 8), dtype=torch.float32, device=device)
    yy = torch.as_tensor(y, dtype=torch.long, device=device)
    weights = np.empty(len(y), np.float32)
    for p in np.unique(person):
        for c in (0, 1, 2):
            mask = (person == p) & (y == c)
            if not mask.any():
                raise ValueError(f"missing training person/class {p}/{c}")
            weights[mask] = 1 / mask.sum()
    ww = torch.as_tensor(weights / weights.mean(), device=device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)
    model = Head(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(y), generator=gen, device=device)
        for rows in order.split(128):
            keep = (torch.rand(xx[rows].shape, generator=gen, device=device) >= 0.1).to(xx.dtype)
            logits = model(xx[rows] * keep)
            loss = (nn.functional.cross_entropy(logits, yy[rows], reduction='none') * ww[rows]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    return model, mu, sd
