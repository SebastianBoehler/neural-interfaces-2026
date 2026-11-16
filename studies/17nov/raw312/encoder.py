"""Independent task encoders with an identical information-preserving readout."""
import copy
import hashlib

import numpy as np
import torch
from torch import nn

BANDS = ((.1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 55))


def moments(x):
    f = torch.fft.rfftfreq(480, 1 / 120).to(x.device)
    p = torch.fft.rfft(x).abs().square() / 480 ** 2
    return torch.cat([x.mean(-1)] + [p[:, :, (f >= lo) & (f < hi)].mean(-1)
                     .clamp_min(1e-12).log() for lo, hi in BANDS], 1)


def fingerprint(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Encoder(nn.Module):
    def __init__(self, arch, seed, device):
        super().__init__()
        import braindecode.models as bd
        torch.manual_seed(seed)
        cls = {'fb': bd.FBCNet, 'sh': bd.ShallowFBCSPNet}[arch]
        self.body = cls(n_chans=41, n_outputs=3, n_times=480, sfreq=120)
        self.body.final_layer = nn.Identity()
        self.to(device).eval()
        with torch.no_grad():
            self.dim = self.body(torch.ones(2, 41, 480, device=device)).flatten(1).shape[1]
        self.task = nn.Linear(self.dim + 287, 3).to(device)
        self.reconstruct = nn.Linear(self.dim, 287).to(device)
        self.register_buffer('moment_mean', torch.zeros(287, device=device))
        self.register_buffer('moment_sd', torch.ones(287, device=device))
        self.frozen = False

    def representation(self, x):
        return self.body(x).flatten(1)

    def forward(self, x):
        z = self.representation(x)
        skip = (moments(x) - self.moment_mean) / self.moment_sd
        return self.task(torch.cat([z, skip], 1))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.body.eval()
        return self


def pretrain(net, raw, seed, epochs=20):
    """Uniform person and record sampling; caller supplies source-only raw rows."""
    gen = torch.Generator(device=raw.device).manual_seed(seed + 312)
    params = [p for p in list(net.body.parameters()) + list(net.reconstruct.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
    with torch.no_grad():
        targets = moments(raw)
        mu, sd = targets.mean(0), targets.std(0).clamp_min(1e-3)
        targets = (targets - mu) / sd
    steps, max_grad, losses = 0, 0., []
    net.train()
    for _ in range(epochs):
        order = torch.randperm(len(raw), generator=gen, device=raw.device)
        for r in order.split(64):
            x = raw[r].clone()
            start = int(torch.randint(0, 361, (1,), generator=gen, device=x.device))
            x[:, :, start:start + 120] = 0
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(net.reconstruct(net.representation(x)), targets[r])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite pretraining loss')
            loss.backward()
            grads = [p.grad for p in net.body.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('inactive/nonfinite encoder gradients')
            max_grad = max(max_grad, max(float(g.abs().max()) for g in grads))
            opt.step(); steps += 1; losses.append(float(loss.detach()))
    if max_grad <= 0:
        raise ValueError('zero pretraining effect')
    return dict(steps=steps, max_gradient=max_grad, first_loss=losses[0], last_loss=losses[-1],
                scaler_sha256=hashlib.sha256(torch.stack([mu, sd]).cpu().numpy().tobytes()).hexdigest())


def supervised(net, x, y, seed, epochs, extra_steps=0, val=None):
    params = [p for p in list(net.body.parameters()) + list(net.task.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    with torch.no_grad():
        z = moments(x); net.moment_mean.copy_(z.mean(0)); net.moment_sd.copy_(z.std(0).clamp_min(1e-3))
    counts = torch.bincount(y, minlength=3).float()
    if (counts <= 0).any():
        raise ValueError('source missing class')
    weights = len(y) / (3 * counts)
    best, state, chosen, logs, steps = float('inf'), None, epochs, [], 0

    def update(r):
        nonlocal steps
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(net(x[r]), y[r], weight=weights)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite supervised loss')
        loss.backward(); opt.step(); steps += 1

    net.train()
    for _ in range(extra_steps):
        update(torch.randint(len(x), (64,), generator=gen, device=x.device))
    for epoch in range(epochs):
        net.train()
        for r in torch.randperm(len(x), generator=gen, device=x.device).split(64):
            update(r)
        if val is not None:
            net.eval()
            with torch.no_grad():
                vx, vy = val
                logits = torch.cat([net(b) for b in vx.split(64)])
                loss = float(nn.functional.cross_entropy(logits, vy))
                ba = float(torch.stack([(logits.argmax(1)[vy == c] == c).float().mean() for c in range(3)]).mean())
            logs.append(dict(epoch=epoch + 1, loss=loss, ba=ba))
            if loss < best:
                best, state, chosen = loss, copy.deepcopy(net.state_dict()), epoch + 1
    if state is not None:
        net.load_state_dict(state)
    return dict(steps=steps, selected_epochs=chosen, source_validation=logs)
