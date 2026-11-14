"""ROBUSTNESS OF THE GATED COUNT FILTER (ledger s238, item 2). Synthetic sessions built from the 14 public sessions
(new people) whose run design differs from the public one; runs stay contiguous and are shuffled inside (within-like),
SEEDS shuffles. The chain's per-window outputs are reused from one order's SAVE_OUT file (SRC; the chain is not rerun
in the new order, so this tests the filter's structure only). The filter is package.py's (a): the 'lag' detector latches
at the first passing check, then J-wprior/urn (hypotheses and urn from the session start, causal) is emitted.
Violations: normal | drop_online | dup_online | online36 | no_acq | acq_last | double | drop10.
Metric: balanced accuracy (%) per synthetic session, filter - no filter, mean over seeds and bases, then per person;
P10's third session (two online runs of 36, a real deviation) is also shown alone. SELFCHECK=1 adds the self-check.
Usage: [SRC=runs/chainout/within0.npz] [ARMS=P0r,V6Fr] [SEEDS=3] [SELFCHECK=0] python robust.py [violation ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import detect as D  # noqa: E402
import structured as S  # noqa: E402
import jointcount as J  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

SRC = os.environ.get("SRC", "runs/chainout/within0.npz"); ARMS = os.environ.get("ARMS", "P0r,V6Fr").split(",")
SEEDS = int(os.environ.get("SEEDS", 3)); SELF = int(os.environ.get("SELFCHECK", 0)); CHECKS = (40, 50, 60, 80, 100, 150)
SC_K, SC_T = int(os.environ.get("SC_K", 30)), float(os.environ.get("SC_T", -3.0))  # self-check: window, log-BF threshold
THR = np.load(C.REPO + "/runs/rundetect/decisions.npz"); WS = list(THR["W"]); X, LAM = C.projected()
VIOL = ("normal", "drop_online", "dup_online", "online36", "no_acq", "acq_last", "double", "drop10")


def runs_of(s):
    rows = np.flatnonzero(C.SES == s); return [rows[C.RUN[rows] == u] for u in dict.fromkeys(C.RUN[rows])]


def build(s, v, rng):
    R = runs_of(s); on = [i for i, r in enumerate(R) if "Online" in C.RUN[r[0]].split("|")[1]]; acq = [i for i in range(len(R)) if i not in on]
    if v == "drop_online":
        k = rng.choice(on); R = [r for i, r in enumerate(R) if i != k]
    elif v == "dup_online":
        k = rng.choice(on); R = R[:k + 1] + [R[k]] + R[k + 1:]
    elif v == "online36":
        for j, i in enumerate(on):
            nxt = R[on[(j + 1) % len(on)]]; R[i] = np.concatenate([R[i]] + [rng.choice(nxt[C.Y[nxt] == c], 2, replace=False) for c in range(3)])
    elif v == "no_acq":
        R = [R[i] for i in on]
    elif v == "acq_last":
        R = [R[i] for i in on] + [R[i] for i in acq]
    elif v == "double":
        R = R + R
    seq = np.concatenate([r[rng.permutation(len(r))] for r in R])
    if v == "drop10":
        seq = seq[np.sort(rng.choice(len(seq), int(round(0.9 * len(seq))), replace=False))]
    return seq


def gated(Pseq, seq, p):
    """-> emitted (T, 3), latch window, self-check off window (or None)."""
    v = D.stat(X, seq, "lag", [w for w in CHECKS if w <= len(seq)]); hit = [w for w, x in zip(CHECKS, v) if x < THR[f"lag|{p}"][WS.index(w)]]
    lw = hit[0] if hit else None
    if lw is None:
        return Pseq.copy(), None, None
    Ls, lp, cap = S.hyps("wprior", p); bounds = np.concatenate([np.zeros((len(Ls), 1), int), np.cumsum(Ls, 1), np.full((len(Ls), 1), 10 ** 6)], 1)
    T = len(seq); blk = np.stack([S.block_index(bounds, t) for t in range(T)]); reset = np.vstack([np.ones(len(Ls), bool), blk[1:] != blk[:-1]])
    capx = np.concatenate([cap, np.full((len(Ls), 2), 12)], 1)[np.arange(len(Ls))[None], blk]
    pr = J.urn(Pseq, reset, capx); ev = np.log(np.maximum(pr.sum(2), 1e-300)); cum = np.vstack([np.zeros(len(lp)), np.cumsum(ev, 0)[:-1]])
    lw_ = lp[None] + cum; w = np.exp(lw_ - logsumexp(lw_, 1, keepdims=True)); o = (w[..., None] * pr).sum(1); o /= o.sum(1, keepdims=True)
    out = Pseq.copy(); out[lw:] = o[lw:]; off = None
    if SELF:  # urn evidence (mixture predictive) against a uniform urn, summed over the last SC_K windows; off for good below SC_T
        L = np.clip(Pseq, 1e-9, 1) ** J.TAU; L /= L.sum(1, keepdims=True)
        bf = np.log((w * pr.sum(2)).sum(1)) - np.log(L.mean(1)); cs = np.concatenate([[0], np.cumsum(bf)])
        for t in range(lw, T):
            if cs[t + 1] - cs[max(t + 1 - SC_K, 0)] < SC_T:
                off = t; out[t:] = Pseq[t:]; break
    return out, lw, off


def bacc(P, y):
    pr = P.argmax(1); return 100 * np.mean([(pr[y == c] == c).mean() for c in np.unique(y)])


def main():
    z = np.load(C.REPO + "/" + SRC); bases = sorted({k.split("|")[0] for k in z.files}); viol = sys.argv[1:] or VIOL; rng0 = np.random.default_rng(0)
    print(f"ROBUST (s238): src {SRC}, {SEEDS} seeds, self-check {'on (K %d, T %g)' % (SC_K, SC_T) if SELF else 'off'}; balanced accuracy change, filter - none, per person mean [80 % bootstrap]")
    for v in viol:
        for arm in ARMS:
            per, p10s7, offs, lat = {}, np.nan, [], []
            for s in C.SESSIONS:
                p = C.PER[np.flatnonzero(C.SES == s)[0]]; ds = []
                for sd in range(SEEDS):
                    seq = build(s, v, np.random.default_rng([sd, int(s), VIOL.index(v)]))
                    for bn in bases:
                        Pq = z[f"{bn}|{arm}"].astype(np.float64)[seq]; out, lw, off = gated(Pq, seq, p)
                        ds.append(bacc(out, C.Y[seq]) - bacc(Pq, C.Y[seq])); lat.append(lw is not None); offs.append(off is not None)
                per.setdefault(p, []).append(np.mean(ds))
                if any("Online" in u.split("|")[1] and (C.RUN == u).sum() == 36 for u in np.unique(C.RUN[C.SES == s])):
                    p10s7 = np.mean(ds)  # P10's third session: two online runs of 36
            d = np.array([np.mean(per[q]) for q in C.PEOPLE]); bt = np.percentile([d[rng0.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
            print(f"  {v:11s} {arm:4s} {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] min person {d.min():+.2f} | latched {np.mean(lat) * 100:.0f} % | self-check off {np.mean(offs) * 100:.0f} % | P10 s3 (online 36) {p10s7:+.2f}", flush=True)


if __name__ == "__main__":
    main()
