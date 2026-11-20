"""Lane Y idea #3: member trained on the chain's own out-of-person errors (registered in evidence/wave_20nov/laneY_research2.md).
python err_member.py OUT.npz [k] : nine-person folds over the ten PROTEUS people; arms W (weight 1 + k*err) and U (unweighted control), same seeds.
err of a training window = mean over orders of (stock terminal wrong), taken from THAT person's own held-out chain run (stored, work/laneZ/*_prof.npz);
the held person's own chain output is never used for its fold. Input: per-cell EA-whitened raw window (unsupervised), compact EEGNet-style net.
Outputs out-of-fold probabilities (recording layout, seed mean) per arm."""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[1]
SEEDS, EPOCHS, KW = (0, 1, 2), 40, 2.0


def ea(X, cells):
    out = np.empty_like(X)
    for c in np.unique(cells):
        m = cells == c; C = np.einsum("nct,ndt->cd", X[m], X[m]) / (m.sum() * X.shape[-1]) + 1e-6 * np.eye(X.shape[1])
        w, V = np.linalg.eigh(C); W = (V / np.sqrt(w)) @ V.T; out[m] = np.einsum("ij,njt->nit", W, X[m])
    return out


class Net(nn.Module):
    def __init__(s, ch=41, f1=8, d=2, T=480):
        super().__init__()
        s.t = nn.Conv2d(1, f1, (1, 31), padding=(0, 15), bias=False); s.b1 = nn.BatchNorm2d(f1)
        s.sp = nn.Conv2d(f1, f1 * d, (ch, 1), groups=f1, bias=False); s.b2 = nn.BatchNorm2d(f1 * d)
        s.sep = nn.Conv2d(f1 * d, f1 * d, (1, 15), padding=(0, 7), groups=f1 * d, bias=False); s.b3 = nn.BatchNorm2d(f1 * d)
        s.dr = nn.Dropout(0.5); s.fc = nn.Linear(f1 * d * (T // 32), 3)

    def forward(s, x):
        x = s.b1(s.t(x[:, None])); x = Fn.avg_pool2d(Fn.elu(s.b2(s.sp(x))), (1, 4)); x = s.dr(x)
        x = Fn.avg_pool2d(Fn.elu(s.b3(s.sep(x))), (1, 8)); return s.fc(s.dr(x.flatten(1)))


def train(X, y, w, Xte, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    Xt, yt, wt = torch.as_tensor(X).to(dev), torch.as_tensor(y).to(dev), torch.as_tensor(w, dtype=torch.float32).to(dev)
    cw = torch.as_tensor(len(y) / (3 * np.bincount(y, minlength=3)), dtype=torch.float32).to(dev)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EPOCHS * ((len(y) + 63) // 64))
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(y), device=dev)
        for i in range(0, len(y), 64):
            ix = perm[i:i + 64]; xb = Xt[ix] * (1 + 0.1 * torch.randn(len(ix), 1, 1, device=dev))
            l = (Fn.cross_entropy(net(xb), yt[ix], reduction="none", weight=cw) * wt[ix]).mean()
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        return torch.cat([Fn.softmax(net(torch.as_tensor(Xte[i:i + 256]).to(dev)), 1).cpu() for i in range(0, len(Xte), 256)]).numpy()


def main(dest, smoke=False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    X, y, subj = z["X"], z["y"], z["subject"]; cells = np.array(["|".join(map(str, t)) for t in zip(subj, z["session"], z["interface"])])
    X = ea(X.astype(np.float64), cells); X = (X / X.std(axis=(1, 2), keepdims=True)).astype(np.float32)
    err = np.zeros(len(y))
    for o in ("full0", "within0"):
        d = np.load(ROOT / f"work/laneZ/{o}_prof.npz"); assert (d["y"] == y).all(); err += (d["stock|base|terminal"].argmax(1) != y) / 2
    out = {a: np.zeros((len(y), 3), np.float32) for a in ("W", "U")}
    people = sorted(set(subj))[: 2 if smoke else None]
    for p in people:
        te, tr = subj == p, subj != p
        for arm in ("W", "U"):
            w = 1 + KW * err[tr] if arm == "W" else np.ones(tr.sum()); w = w / w.mean()
            out[arm][te] = np.mean([train(X[tr], y[tr], w, X[te], s, dev) for s in (SEEDS[:1] if smoke else SEEDS)], 0)
        print(p, {a: float((out[a][te].argmax(1) == y[te]).mean()) for a in out}, flush=True)
    np.savez(dest, y=y, err=err, **out)


if __name__ == "__main__":
    main(sys.argv[1], smoke=len(sys.argv) > 2)
