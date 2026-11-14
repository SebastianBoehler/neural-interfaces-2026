"""RUNDETECT boundaries and blocks (ledger s217, tasks 2-3). In delivery order, per session, on the nuisance projection
(common.projected: leave-one-person-out, within-run SD 1, between-run variance LAM from the other people):
  bocpd  online Bayesian change point: state = length of the current run, hazard from the other people's run lengths,
         conjugate Gaussian run means around the causal session mean. A window's label is fixed when it arrives: a new
         label starts when the posterior mass on a run start after the current label's start exceeds THETA (causal).
  cnt    the same filter with the features ignored (length prior only): what the features add.
  dpoff  offline length-constrained segmentation (exact DP, whole session): non-causal reference only, never a gate.
  mix    online sequential mixture over <= 6 blocks (full shuffle): capacity-weighted prior, conjugate means, hard
         MAP assignment when the window arrives (causal).
  loopW  detector in the loop: bocpd labels before window W; from W bocpd if the 'lag' detector (thresholds fitted on
         the other people, detect.py) says contiguous at W, else singletons (the count filter is then inert).
  loopMW the same, but mix labels (offset) instead of singletons from W when the detector says not contiguous.
Writes runs/runids/<name>.npz (key '<scheme><seed>' -> (N,) label per replay row, recording layout, unique within a
session) for every order of shuffle_perms (+ 'recording'), and prints purity / ARI against the true runs per scheme.
Usage: python studies/14nov/rundetect/segment.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import detect as D  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

ORDERS = D.ORDERS; HI = 44; CAPB, KMAX = 36, 6
LOOPS = (40, 60); THETA = float(os.environ.get("THETA", 0.5))  # new label when P(run start after the current label's start) > THETA


def gauss(x, m, v):  # log N(x; m, diag v), summed over directions
    return -0.5 * (np.log(2 * np.pi * v) + (x - m) ** 2 / v).sum(-1)


def hazard(pl):
    surv = pl[::-1].cumsum()[::-1]; h = np.where(surv > 0, pl / np.maximum(surv, 1e-300), 1.0); h[HI] = 1.0; return h  # h[l]: run ends at length l


def bocpd(X, lam, h, use_x=True):
    n = len(X); lab = np.zeros(n, int); il = 1.0 / lam
    lp = np.array([0.0]); S = X[:1].copy(); cur_start, cur = 0, 0  # hypotheses: current run length l = 1..len(lp)
    for t in range(1, n):
        mu0 = X[:t].mean(0); L = np.arange(1, len(lp) + 1)
        pm = (mu0 * il + S) / (il + L[:, None]); pv = 1.0 / (il + L[:, None]) + 1.0
        ll_c = gauss(X[t], pm, pv) if use_x else np.zeros(len(L)); ll_n = gauss(X[t], mu0, lam + 1.0) if use_x else 0.0
        hz = h[np.minimum(L, HI)]
        grow = lp + np.log(np.maximum(1 - hz, 1e-300)) + ll_c; new = logsumexp(lp + np.log(np.maximum(hz, 1e-300))) + ll_n
        lp = np.concatenate([[new], grow]); lp -= logsumexp(lp); S = np.concatenate([X[t:t + 1], S + X[t]])
        keep = np.arange(len(lp)) < HI; lp, S = lp[keep], S[keep]; lp -= logsumexp(lp)
        post = np.exp(lp); after = post[:t - cur_start].sum()  # mass on a run start after the current label's start
        if after > THETA:
            cur += 1; cur_start = t - int(np.argmax(post[:t - cur_start]))
        lab[t] = cur
    return lab


def dpoff(X, lam, pl):
    n, mu0 = len(X), X.mean(0); cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)]); cq = np.vstack([np.zeros(X.shape[1]), np.cumsum(X ** 2, 0)])
    lpl = np.log(np.maximum(pl, 1e-300)); best = np.full(n + 1, -np.inf); best[0] = 0; arg = np.zeros(n + 1, int)
    for b in range(1, n + 1):
        for ln in range(1, min(HI, b) + 1):
            a = b - ln
            if not np.isfinite(best[a]) or pl[ln] <= 0:
                continue
            s, q = cs[b] - cs[a], cq[b] - cq[a]; xb = s / ln
            ml = (-0.5 * ln * np.log(2 * np.pi) - 0.5 * np.log(1 + ln * lam) - 0.5 * ((q - s * xb) + ln * (xb - mu0) ** 2 / (1 + ln * lam))).sum()
            v = best[a] + lpl[ln] + ml
            if v > best[b]:
                best[b], arg[b] = v, a
    lab = np.zeros(n, int); b, k = n, 0
    while b > 0:
        lab[arg[b]:b] = k; b = arg[b]; k += 1
    return lab.max() - lab


def mix(X, lam):
    n = len(X); lab = np.zeros(n, int); il = 1.0 / lam; S, cnt = [X[0].copy()], [1]
    for t in range(1, n):
        mu0 = X[:t].mean(0); lw = []
        for k in range(len(cnt)):
            pm = (mu0 * il + S[k]) / (il + cnt[k]); pv = 1.0 / (il + cnt[k]) + 1.0; lw.append(np.log(max(CAPB - cnt[k], 0) + 0.5) + gauss(X[t], pm, pv))
        if len(cnt) < KMAX:
            lw.append(np.log(33.0 * (KMAX - len(cnt))) + gauss(X[t], mu0, lam + 1.0))
        k = int(np.argmax(lw))
        if k == len(cnt):
            S.append(X[t].copy()); cnt.append(1)
        else:
            S[k] = S[k] + X[t]; cnt[k] += 1
        lab[t] = k
    return lab


def main():
    X, LAM = C.projected(); thr = np.load(C.REPO + "/runs/rundetect/decisions.npz"); WS = list(thr["W"])
    names = ["bocpd", "cnt", "dpoff", "mix"] + [f"loop{m}{w}" for w in LOOPS for m in ("", "M")]; out = {nm: {} for nm in names}; met = {}
    for o in ORDERS:
        lab = {nm: np.full(C.N, -1) for nm in names}
        for s in C.SESSIONS:
            rows = C.order_rows(o, s); p = C.PER[rows[0]]; v = X[rows]; pl = C.length_prior(p, hi=HI); h = hazard(pl)
            seg = {"bocpd": bocpd(v, LAM[p], h), "cnt": bocpd(v, LAM[p], h, use_x=False), "dpoff": dpoff(v, LAM[p], pl), "mix": mix(v, LAM[p])}
            for w in LOOPS:
                contig = D.stat(X, rows, "lag", [w])[0] < thr[f"lag|{p}"][WS.index(w)]
                seg[f"loop{w}"] = seg["bocpd"].copy() if contig else np.concatenate([seg["bocpd"][:w], 1000 + np.arange(len(rows) - w)])
                seg[f"loopM{w}"] = seg["bocpd"].copy() if contig else np.concatenate([seg["bocpd"][:w], 1000 + seg["mix"][w:]])
            for nm in names:
                lab[nm][rows] = seg[nm]; met.setdefault((nm, D.scheme(o)), []).append(C.purity_ari(seg[nm], C.RUNI[rows]) + (len(np.unique(seg[nm])),))
        for nm in names:
            assert (lab[nm] >= 0).all(); out[nm][o] = lab[nm]
    for nm in names:
        np.savez(C.REPO + f"/runs/runids/{nm}.npz", **out[nm])
    print(f"RUNDETECT boundaries (s217): NDIM {C.NDIM} REG {C.REG:g}; purity / ARI / labels per session vs true runs, mean over sessions x seeds")
    for nm in names:
        print(f"  {nm:7s} " + " | ".join(f"{sc} {np.mean([m[0] for m in met[nm, sc]]):.2f} / {np.mean([m[1] for m in met[nm, sc]]):.2f} / {np.mean([m[2] for m in met[nm, sc]]):.1f}" for sc in ("recording", "runs", "within", "full")))


if __name__ == "__main__":
    main()
