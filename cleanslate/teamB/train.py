"""Team B train.py (harness contract): python train.py --people P01,... [--first-session-only P04] --out DIR

Trains the encoder ensemble (families x seeds, CSB_NETS e.g. "shallow:3+slow:3+fbc:3"; "+" or ",") on the permitted PROTEUS windows
and writes DIR/submission.py + DIR/weights.pt. Each batch holds G sessions x 3*PER class-balanced windows; every group is
aligned by a random-size subset of its own session (mimics the causal prefix), standardised over the group (SFN) and
trained with cross-entropy plus a within-session supervised-contrastive loss on the projection.
"""
import importlib.util
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
import data as D  # noqa: E402  harness/data.py


def load_sub():
    spec = importlib.util.spec_from_file_location("teamB_submission", HERE / "submission.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    return sub


sub = load_sub()
G, PER = 6, 8
STEPS = int(os.environ.get("CSB_STEPS", 1200))
LAM = float(os.environ.get("CSB_LAM", 0.5))
TAU = 0.2


def supcon(q, y):
    """Within-group supervised contrastive loss. q (G,n,PD) unit, y (G,n)."""
    n = q.shape[1]
    sim = q @ q.transpose(1, 2) / TAU
    eye = torch.eye(n, dtype=torch.bool, device=q.device)
    logp = sim.masked_fill(eye, -1e9).log_softmax(-1)
    pos = (y[:, :, None] == y[:, None, :]) & ~eye
    return -(logp * pos).sum(-1).div(pos.sum(-1).clamp_min(1)).mean()


class Bank:
    """Training windows on the GPU: detrended signals, broadband covariances, session and class index lists."""

    def __init__(self, X, y, session_key, person, dev):
        self.X = torch.cat([sub.detrend(torch.from_numpy(X[i:i + 256]).to(dev)) for i in range(0, len(X), 256)])
        mask = sub.fmask(1.0, 45.0).to(dev)
        xb = torch.cat([sub.bandpass(self.X[i:i + 256], mask) for i in range(0, len(X), 256)]).double()
        self.cov = xb @ xb.transpose(1, 2) / sub.T
        self.y = torch.from_numpy(y).to(dev)
        self.people = sorted(set(person))
        self.sess = {p: sorted(set(session_key[person == p])) for p in self.people}
        self.rows = {s: torch.from_numpy(np.flatnonzero(session_key == s)).to(dev) for s in set(session_key)}
        self.cls = {s: [r[self.y[r] == c] for c in range(3)] for s, r in self.rows.items()}
        assert all(len(c) >= PER for v in self.cls.values() for c in v), "a session lacks windows of a class"

    def batch(self, g):
        xs, ys = [], []
        for _ in range(G):
            p = self.people[int(torch.randint(len(self.people), (1,), generator=g))]
            s = self.sess[p][int(torch.randint(len(self.sess[p]), (1,), generator=g))]
            rows, n = self.rows[s], len(self.rows[s])
            m = int(round(math.exp(float(torch.rand(1, generator=g)) * math.log(n / 4)) * 4))  # log-uniform in [4, n]
            ref = rows[torch.randperm(n, generator=g)[:m].to(rows.device)]
            W = sub.whitener(self.cov[ref].sum(0), m).float()
            idx = torch.cat([c[torch.randperm(len(c), generator=g)[:PER].to(c.device)] for c in self.cls[s]])
            xs.append(W @ self.X[idx])
            ys.append(self.y[idx])
        x = torch.cat(xs)
        return torch.roll(x, int(torch.randint(-24, 25, (1,), generator=g)), -1), torch.stack(ys)


def train_net(kind, seed, bank, dev, steps=STEPS):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = sub.Net(kind).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    k, hist = sub.DEFAULTS["sfn_k"], []
    for step in range(steps):
        x, y = bank.batch(g)
        z = sub.sfn(net.feats(x).view(G, 3 * PER, -1), k)
        logits, q = net.heads(z)
        l_ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), label_smoothing=0.1)
        l_sc = supcon(q, y)
        opt.zero_grad()
        (l_ce + LAM * l_sc).backward()
        opt.step()
        sched.step()
        hist.append((l_ce.item(), l_sc.item()))
    ce, sc = (sum(h[j] for h in hist[-100:]) / len(hist[-100:]) for j in (0, 1))
    assert math.isfinite(ce) and math.isfinite(sc), "training diverged"
    return {k_: v.detach().cpu() for k_, v in net.eval().state_dict().items()}, ce, sc


def fit(people, first_only, out, data=None, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = os.environ.get("CSB_NETS", "shallow:3,slow:3,fbc:3")
    X, y, info = D.load_train(people, first_only, data)
    key = np.char.add(np.char.add(info["person"], "|"), info["session"])
    bank = Bank(X, y, key, info["person"], dev)
    print(f"[teamB] train {len(y)} windows, {len(bank.rows)} sessions, people {','.join(bank.people)}, nets {spec}, "
          f"steps {STEPS}, lam {LAM}, device {dev}", flush=True)
    nets = []
    for item in spec.replace("+", ",").split(","):
        kind, count = item.split(":")
        assert kind in sub.BODIES, kind
        for j in range(int(count)):
            t0 = time.time()
            state, ce, sc = train_net(kind, 1000 * seed + 17 * len(nets) + j, bank, dev)
            nets.append((kind, state))
            print(f"[teamB] {kind}#{j} ce {ce:.3f} supcon {sc:.3f} {time.time() - t0:.0f}s", flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(nets=nets, cfg={}, people=list(people), first_only=list(first_only), spec=spec, steps=STEPS,
                    lam=LAM), out / "weights.pt")
    shutil.copy(HERE / "submission.py", out / "submission.py")


if __name__ == "__main__":
    a = D.train_parser("team B")
    fit(a.people, a.first_session_only, a.out, a.data, a.seed)
