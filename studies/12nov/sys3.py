"""S3 (ledger s98): a causal session-set transformer that learns the session adaptation, trained on synthetic sessions.
Inputs: runs/sys3_inputs.npz (per fold q: X_q (n, d) = S2's standardised multi-view features; y, person, sess).
Usage (Galvani): python studies/12nov/sys3.py TASK   (fold = TASK % 10, seed = TASK // 10) | smoke"""
import sys, time
from pathlib import Path
import numpy as np
import torch
WS = Path(__file__).resolve().parents[2]; OUT = WS / "runs/diag/sys3"; DEV = "cuda" if torch.cuda.is_available() else "cpu"
Dd = np.load(WS / "runs/sys3_inputs.npz", allow_pickle=True); y, per, sess = Dd["y"].astype(int), Dd["person"].astype(str), Dd["sess"].astype(str)
PEOPLE = sorted(set(per)); SID = {s: np.flatnonzero(sess == s) for s in np.unique(sess)}


class Model(torch.nn.Module):
    def __init__(s, d, h=96):
        super().__init__(); s.inp = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(0.3)); s.pos = torch.nn.Linear(1, h)
        layer = torch.nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.2, batch_first=True, norm_first=True); s.enc = torch.nn.TransformerEncoder(layer, 2); s.out = torch.nn.Linear(h, 3)

    def forward(s, x):  # x (1, n, d) -> (n, 3); causal: window t attends to windows <= t
        n = x.shape[1]; t = torch.arange(n, device=x.device, dtype=torch.float32)[None, :, None] / 100.0
        h = s.inp(x) + s.pos(t); mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
        return s.out(s.enc(h, mask=mask))[0]


def episode(X, r, rng):
    """synthetic session: random class-share shift, random subset (order kept), random rotation + shift + scale of the features"""
    w = rng.dirichlet(np.ones(3) * 4)[y[r]]; keep = rng.random(len(r)) < np.clip(0.6 + 0.8 * w, 0.3, 1.0); r = r[keep]
    x = X[r]; d = x.shape[1]; Q, _ = np.linalg.qr(np.eye(d) + 0.15 / np.sqrt(d) * rng.standard_normal((d, d)))  # 12 Nov fix: 0.15 per entry made Q a random rotation
    assert np.abs(np.diag(Q)).mean() > 0.8, "the rotation is not small"
    x = (x @ Q) * rng.uniform(0.8, 1.25, d) + 0.3 * rng.standard_normal(d); return x.astype(np.float32), y[r]


def run(task, smoke=False):
    q = PEOPLE[task % 10]; seed = task // 10; X = Dd[f"X_{q}"]; rng = np.random.default_rng(seed); torch.manual_seed(seed)
    train = [r for s, r in SID.items() if per[r[0]] != q]; test = [r for s, r in SID.items() if per[r[0]] == q]
    m = Model(X.shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-2); E = 50 if smoke else 20000; t0 = time.time()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=E)
    for e in range(E):
        x, yy = episode(X, train[rng.integers(len(train))], rng); lo = m(torch.from_numpy(x)[None].to(DEV))
        loss = torch.nn.functional.cross_entropy(lo[3:], torch.from_numpy(yy[3:]).to(DEV), label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if e % 2000 == 0:
            print(f"{q} s{seed} episode {e} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
    m.eval(); P = np.zeros((len(y), 3)); rows = np.concatenate(test)
    with torch.no_grad():
        for r in test:
            P[r] = torch.softmax(m(torch.from_numpy(X[r].astype(np.float32))[None].to(DEV)), 1).cpu().numpy()
    acc = np.mean(P[rows].argmax(1) == y[rows]); OUT.mkdir(parents=True, exist_ok=True)
    if not smoke:
        np.savez(OUT / f"sys3_{q}-s{seed}.npz", proba=P[rows], index=rows)
    print(f"TOTAL {q} s{seed} window accuracy {acc:.4f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(0, smoke=True) if sys.argv[1] == "smoke" else run(int(sys.argv[1]))
