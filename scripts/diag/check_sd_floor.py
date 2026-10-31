"""Bug-fix check (31 Oct): SessionEM sd floor 1e-3 vs the old sd + 1e-9 (R19 and R19e views).
Gate for keeping the fix: pooled six-seed dla and ds not worse than -0.3 on either seed half."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_common import *  # d, load, mixp, rebal, cm
import session_em as SE

X = d["X"].astype(np.float64); ch = list(d["ch_names"]); st = d["stream"]
G = SE.gaze_features(X, ch)
EV = SE.event_features(X, ch, st, SE.global_mad(X, ch))
del X
FROZEN = dict(lam=0.5, n0=60, m0=20, shrink=0.7, iters=2, nmin=10)


def _old(self, A, B, f, p, t):
    m, sd = A.mean(0), A.std(0) + 1e-9
    Z, z = (A - m) / sd, (f - m) / sd
    Q = B
    for _ in range(self.iters):
        mu = (Q / (Q.sum(0, keepdims=True) + self.m0)).T @ Z
        C = np.cov((Z - Q @ mu).T)
        C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
        Ci = np.linalg.inv(C)
        Q = B * SE._softmax_lda(Z, mu, Ci) ** self.lam
        Q = Q / Q.sum(1, keepdims=True)
    e = self.lam * t / (t + self.n0)
    q = p * SE._softmax_lda(z[None], mu, Ci)[0] ** e
    return q / q.sum()


class Old(SE.SessionEM):
    _adapt = _old


def run(cls, F, P, idx):
    out = np.empty_like(P)
    for s in dict.fromkeys(st[idx]):
        em, rows = cls(**FROZEN), np.flatnonzero(st[idx] == s)
        for r in rows:
            out[r] = em.step(F[idx][r], P[r])
    return out


def spec(a, b, tfb):
    return ["v2", f"fbcnet_ea_tf:{a}-{tfb}*0.5", f"fbcnet_raw_tf:{a}-{tfb}*0.5"] + [f"{n}:{a}-{b}" for n in (
        "shallow_ea", "shallow_raw", "fbcnet_raw", "fbcnet_ea_rot05", "sincshallow_ea_ptm7r_rot05", "fblight_ea_rot05")]


for tag, sp in (("s0-2", spec(0, 2, 1)), ("s3-5", spec(3, 5, 4)), ("all6", None)):
    for proto in ("dev_loso_all", "dev_session"):
        _, Ps, w, idx, _ = load(proto, sp)
        base = rebal(mixp(Ps, w), idx)
        line = f"{tag} {proto:13s}"
        for name, F in (("R19", G), ("R19e", np.c_[G, EV])):
            o, n = run(Old, F, base, idx).argmax(1), run(SE.SessionEM, F, base, idx).argmax(1)
            line += f" | {name} old {cm(idx, o):.2f} new {cm(idx, n):.2f} d {cm(idx, n) - cm(idx, o):+.2f} changed {(o != n).sum()}"
        print(line, flush=True)
