"""Lane V: virtual-subject training augmentation for the raw FBCNet SessionBN trunk family (arm B recipe).

The windows a SessionNet batch reads (X[j], EA-aligned, 2 chunks of 32 same-session windows) pass through VX.__getitem__:
  mix  (V1): each window, with prob 0.5, is replaced by lam*x + (1-lam)*x' with x' a random SAME-CLASS window of a DIFFERENT
             training person, lam ~ U[0.5, 1]. Hard label unchanged (same class).
  virt (V2): one random 'person' per 32-window chunk: M = D Q S applied to the chunk, then rescaled to its original mean power.
             S electrode shift: I + K(delta) - K(0), K row-stochastic inverse-distance kernel on the 10-10 grid, delta ~ N(0, shift^2) per axis;
             Q rotation expm(A - A^T), A ~ N(0, rot^2); D channel gain exp(gain * N(0,1)).
V3 = mix then virt."""
import numpy as np
import torch

# approximate scalp grid (x: left -, y: anterior +) for the 41 PROTEUS channels
ROW = {"AF": 5, "F": 4, "FC": 3, "C": 2, "CP": 1, "TP": 1, "P": 0, "PO": -1}
ARMS = {  # arm -> (mix, virt dose: (rot, gain, shift) or None)
    "V0": (False, None), "V1": (True, None), "V2": (False, (0.15, 0.10, 0.35)),
    "V2s": (False, (0.30, 0.20, 0.70)), "V3": (True, (0.15, 0.10, 0.35))}


def grid(names):
    pos = []
    for n in names:
        n = str(n); k = next(i for i, ch in enumerate(n) if ch.isdigit() or ch == "z")
        row, c = n[:k], n[k:]
        x = 0.0 if c == "z" else (-(int(c) + 1) / 2 if int(c) % 2 else int(c) / 2)
        pos.append(((-1.5 if int(c) % 2 else 1.5) if row == "AF" else x, ROW[row]))
    return torch.tensor(pos, dtype=torch.float32)


def kernel(pos, delta, eps=0.1):
    d2 = ((pos[None] + delta[:, None])[:, :, None] - pos[None, None]).pow(2).sum(-1)  # (B,C,C): shifted sensor c to true sensor c'
    k = 1.0 / (d2 + eps)
    return k / k.sum(-1, keepdim=True)


class VX:
    """Indexable stand-in for the training tensor X inside SessionNet._train."""

    def __init__(self, X, y, person, arm, seed, dev, pos, chunk=32):
        self.mix, self.virt = ARMS[arm]
        self.X, self.y, self.person, self.dev, self.chunk = X.to(dev), y.numpy(), person, dev, chunk
        self.pos, self.rng = pos.to(dev), np.random.default_rng(seed + 31337)
        self.g = torch.Generator(dev).manual_seed(seed + 31337)
        self.pool = {c: np.flatnonzero(self.y == c) for c in np.unique(self.y)}
        self.stats = dict(calls=0, mixed=0, windows=0, change=0.0)

    def __len__(self):
        return len(self.X)

    def partners(self, j):
        out = np.empty(len(j), int)
        for c, p in self.pool.items():
            m = np.flatnonzero(self.y[j] == c)
            if not len(m):
                continue
            r = self.rng.choice(p, len(m))
            for _ in range(60):  # redraw same-person partners
                bad = self.person[r] == self.person[j[m]]
                if not bad.any():
                    break
                r[bad] = self.rng.choice(p, int(bad.sum()))
            assert not (self.person[r] == self.person[j[m]]).any(), "mixup partner from the same person"
            out[m] = r
        return out

    def transforms(self, nb, C):
        rot, gain, shift = self.virt
        A = rot * torch.randn(nb, C, C, generator=self.g, device=self.dev)
        Q = torch.linalg.matrix_exp(A - A.transpose(1, 2))
        D = torch.exp(gain * torch.randn(nb, C, generator=self.g, device=self.dev))
        delta = shift * torch.randn(nb, 2, generator=self.g, device=self.dev)
        z = torch.zeros(1, 2, device=self.dev)
        S = torch.eye(C, device=self.dev) + kernel(self.pos, delta) - kernel(self.pos, z)
        return D[:, :, None] * (Q @ S)

    def __getitem__(self, j):
        j = np.asarray(j)
        x0 = self.X[torch.from_numpy(j).to(self.dev)]
        x = x0
        if self.mix:
            sel = torch.from_numpy(self.rng.random(len(j)) < 0.5).to(self.dev)
            lam = torch.from_numpy(0.5 + 0.5 * self.rng.random(len(j))).float().to(self.dev)
            xp = self.X[torch.from_numpy(self.partners(j)).to(self.dev)]
            lam = torch.where(sel, lam, torch.ones_like(lam))[:, None, None]
            x = lam * x + (1 - lam) * xp
            x = x * (x0.pow(2).mean((1, 2), keepdim=True) / x.pow(2).mean((1, 2), keepdim=True)).sqrt()  # keep the window's power
            self.stats["mixed"] += int(sel.sum())
        if self.virt:
            assert len(j) % self.chunk == 0, "batch is not whole chunks"
            nb, C, T = len(j) // self.chunk, x.shape[1], x.shape[2]
            M = self.transforms(nb, C)
            xb = x.reshape(nb, self.chunk, C, T)
            y = torch.einsum("bcd,bkdt->bkct", M, xb)
            y = y * (xb.pow(2).mean((1, 2, 3), keepdim=True) / y.pow(2).mean((1, 2, 3), keepdim=True)).sqrt()
            x = y.reshape(len(j), C, T)
        self.stats["calls"] += 1; self.stats["windows"] += len(j); self.stats["change"] += float((x - x0).abs().mean())
        return x


def install(net, arm, seed, d, ch_names):
    """Wrap net._train so its X is a VX; fit() fails loudly if the augmentation never ran or changed nothing."""
    mix, virt = ARMS[arm]
    if not (mix or virt):
        return None
    holder = {}
    orig, orig_fit = net._train, net.fit

    def train(X, y, *a, **k):
        person = np.unique(net.d["subject"][net._order], return_inverse=True)[1]
        assert len(person) == len(X) == len(net._order)
        holder["vx"] = VX(X, y, person, arm, seed, net.dev, grid(ch_names))
        return orig(holder["vx"], y, *a, **k)

    def fit(idx):
        orig_fit(idx)
        s = holder["vx"].stats
        assert s["calls"] > 0 and s["change"] > 0, f"augmentation never changed a window: {s}"
        assert not mix or s["mixed"] > 0, f"mixup never applied: {s}"
        net.vstats = s

    net._train, net.fit = train, fit
    return holder
