"""OCUNET (ledger s58, step 3): a learned eye-movement branch. Derived ocular traces -> 1-D ResNet with channel attention,
Mixup; leave one person out and the seen protocol. Saves proba, index and the 128-d embedding per (arm, protocol, seed).
Usage (Leonardo or Galvani, NI26_WS): python ocunet.py TASK   (0-5: arm = ("mix", "nomix")[TASK % 2], seed = TASK // 2) | python ocunet.py smoke
"""
import os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
import protocol  # noqa: E402

OUT = WS / "runs/diag/ocunet/probs"; ARMS = ("mix", "nomix"); DEV = "cuda" if torch.cuda.is_available() else "cpu"


def traces(d):
    """(N, 7, 480): two horizontal pairs, two vertical differences, magnitude, two velocities; each trace is centred
    and scaled by the running statistics of its session up to and including the window (causal, label-free)."""
    X = d["X"].astype(np.float64); ix = {str(c): i for i, c in enumerate(d["ch_names"])}; sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)  # noqa: E731
    sg = lambda a: savgol_filter(a, 25, 3, axis=-1)  # noqa: E731
    h1, h2 = sg(sel(("F5", "AF3", "F3")) - sel(("F6", "AF4", "F4"))), sg(sel(("FC5", "C5")) - sel(("FC6", "C6")))
    v1, v2 = sg(sel(("AF3", "AF4")) - sel(("P1", "P2", "Pz"))), sg(sel(("AF3", "AF4")) - sel(("C1", "C2", "Cz")))
    T = np.stack([h1, h2, v1, v2, np.sqrt(h1 ** 2 + v1 ** 2), np.gradient(h1, axis=-1), np.gradient(v1, axis=-1)], 1)
    st = d["stream"]
    for s in dict.fromkeys(st):
        r = np.flatnonzero(st == s); assert (np.diff(r) > 0).all(); n = np.arange(1, len(r) + 1)[:, None]
        m1 = np.cumsum(T[r].mean(-1), 0) / n; m2 = np.cumsum((T[r] ** 2).mean(-1), 0) / n
        T[r] = (T[r] - m1[:, :, None]) / np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))[:, :, None]
    return np.clip(T, -8, 8).astype(np.float32)


class ECA(nn.Module):
    def __init__(self):
        super().__init__(); self.c = nn.Conv1d(1, 1, 3, padding=1, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.c(x.mean(-1, keepdim=True).transpose(1, 2)).transpose(1, 2))


class Block(nn.Module):
    def __init__(self, a, b, stride):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(a, b, 7, stride, 3, bias=False), nn.BatchNorm1d(b), nn.SiLU(), nn.Conv1d(b, b, 7, 1, 3, bias=False), nn.BatchNorm1d(b), ECA())
        self.s = nn.Identity() if (a == b and stride == 1) else nn.Sequential(nn.Conv1d(a, b, 1, stride, bias=False), nn.BatchNorm1d(b))

    def forward(self, x):
        return torch.nn.functional.silu(self.f(x) + self.s(x))


class Net(nn.Module):
    def __init__(self, cin=7):
        super().__init__()
        self.a, self.b = nn.Conv1d(cin, 16, 7, 2, 3, bias=False), nn.Conv1d(cin, 16, 31, 2, 15, bias=False)  # short and long kernels
        self.bn = nn.BatchNorm1d(32); self.body = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 64, 2), Block(64, 128, 2))
        self.drop, self.fc = nn.Dropout(0.3), nn.Linear(128, 3)

    def embed(self, x):
        return self.body(torch.nn.functional.silu(self.bn(torch.cat([self.a(x), self.b(x)], 1)))).mean(-1)

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


def fit_predict(T, y, tr, te, seed, mix, epochs=60):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); net = Net().to(DEV); Xt, yt = torch.from_numpy(T[tr]), torch.from_numpy(y[tr])
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); nb = int(np.ceil(len(tr) / 64)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * nb)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(tr), generator=g)
        for b in range(nb):
            j = perm[b * 64:(b + 1) * 64]
            if len(j) < 8:
                sched.step(); continue
            xb, yb = Xt[j].to(DEV), torch.nn.functional.one_hot(yt[j], 3).float().to(DEV)
            xb = xb * (0.8 + 0.4 * torch.rand(len(j), 1, 1, generator=g).to(DEV)) + 0.05 * torch.randn(xb.shape, generator=g).to(DEV)
            if mix:
                lam = float(np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,), generator=g))).beta(0.2, 0.2)); k = torch.randperm(len(j), generator=g).to(DEV)
                xb, yb = lam * xb + (1 - lam) * xb[k], lam * yb + (1 - lam) * yb[k]
            opt.zero_grad(); loss = -(yb * torch.log_softmax(net(xb), 1)).sum(1).mean(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        xe = torch.from_numpy(T[te]).to(DEV); E = torch.cat([net.embed(xe[i:i + 512]) for i in range(0, len(te), 512)]); P = torch.softmax(net.fc(E).double(), 1)
    return P.cpu().numpy(), E.cpu().numpy().astype(np.float32)


def run(arm, d, T, proto, seed, out, epochs=60):
    f = out / f"oc_{arm}-{proto}-s{seed}.npz"
    if f.exists():
        return print("exists", f.name, flush=True)
    idx, P, E, t0 = [], [], [], time.time()
    for fold, tr, te in protocol.folds(d, proto):
        p, e = fit_predict(T, d["y"], np.flatnonzero(tr), np.flatnonzero(te), seed, arm == "mix", epochs); idx.append(np.flatnonzero(te)); P.append(p); E.append(e)
        print(f"{arm} s{seed} {proto} {fold} acc {np.mean(p.argmax(1) == d['y'][np.flatnonzero(te)]):.3f} {time.time() - t0:.0f}s", flush=True)
        if epochs < 60:
            break
    idx, P, E = np.concatenate(idx), np.concatenate(P), np.concatenate(E); o = np.argsort(idx)
    np.savez(f, proba=P[o], index=idx[o], emb=E[o]); print(f"TOTAL {f.name} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load(); T = traces(d)
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/ocunet/smoke"; out.mkdir(parents=True, exist_ok=True); run("mix", d, T, "seen_session", 0, out, epochs=2); return print("SMOKE PASS", flush=True)
    t = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        run(ARMS[t % 2], d, T, proto, t // 2, OUT)


if __name__ == "__main__":
    main()
