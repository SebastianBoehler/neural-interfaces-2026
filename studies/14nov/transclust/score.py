"""BALANCED TRANSDUCTIVE CLUSTERING scorer (ledger s259 incl. the 14 Nov 18:19 amendment). r15 chain (runs/chainout/s242r15_<order>
V6Fr, six bases), chain posterior for the mapping = plabels_r15.npz '<order>|raw'. Arms: sess (primary, per-session balance),
run (oracle, true run ids). Cluster -> class: Hungarian on the chain posterior summed per cluster, per session and order.
 Non-causal: purity of the mapped cluster labels vs the chain's argmax on the same windows; (a) chain x soft cluster posterior
 (Sinkhorn plan, weight 1); (b) aE (classic.py) trained on the mapped cluster labels of the four training runs, fused chain x aE.
 Causal prefix (a): in the delivery order of runs/shuffle_perms.npz '<scheme>|<i>|all_loso', cluster the first t windows of each
 session (per-session balance), map by the chain posterior of those windows, score the windows from t on (nearest centroid; soft =
 row-normalised exp(-(d - dmin) / (EPS x median prefix distance))); chain and fusion scored on the same later windows.
Usage: python studies/14nov/transclust/score.py [smoke] > evidence/transclust_14nov/score.txt"""
import sys
from itertools import permutations
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[0] / "personal_ceiling")]
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
import classic as CL  # noqa: E402
import splits as S  # noqa: E402
import space as SP  # noqa: E402
from score_pl import ALL, PAIRS, boot, wmean  # noqa: E402

R = S.REPO; m = S.meta(); Y = m["y"]; SMOKE = "smoke" in sys.argv; ORD = ALL[:2] if SMOKE else ALL; TS = (80, 120)
PL = np.load(S.OUT / "plabels_r15.npz"); CLU = np.load(R / "runs/transclust/clusters.npz"); PERM = np.load(R / "runs/shuffle_perms.npz")
CH = {o: [np.load(R / f"runs/chainout/s242r15_{o}.npz")[k].astype(np.float64) for k in sorted(np.load(R / f"runs/chainout/s242r15_{o}.npz").files) if k.endswith("|V6Fr")] for o in ORD}
assert all(len(v) == 6 for v in CH.values()); TEST = {(s, n): t for s, n, t in PAIRS}; PEOP = lambda s: S.PEOPLE.index(s[:3])  # noqa: E731


def mapping(P, lab):
    """cluster -> class by Hungarian on the chain posterior summed per cluster"""
    M = np.array([P[lab == k].sum(0) for k in range(3)]); a, b = linear_sum_assignment(-M); mp = np.zeros(3, int); mp[a] = b; return mp


def soft_classes(Q, mp):
    out = np.zeros_like(Q); out[:, mp] = Q; return out


def by_person(vals):
    """vals: {session: x} -> (10,) person means over sessions"""
    return np.array([np.nanmean([v for s, v in vals.items() if s.startswith(p)]) for p in S.PEOPLE])


def fused_gain(o, s, rows_of, Q, base_only=False):
    """mean over the session's pairs and bases of cell-score(chain x Q) - cell-score(chain) on rows_of(test rows)"""
    g = []
    for n in [n for ss, n in TEST if ss == s]:
        t = rows_of(TEST[s, n])
        if len(t):
            g.append(np.mean([S.cell_score(m, t, S.fuse(c[t], Q[t])) - S.cell_score(m, t, c[t]) for c in CH[o]]))
    return np.mean(g) if g else np.nan  # no test window delivered after t (causal arms)


def noncausal(arm):
    """-> purity arrays and gain arrays (orders, 10) for arm; also caches aE fits by (session, pair, mapping)"""
    hard, soft = CLU[f"hard_{arm}"], CLU[f"soft_{arm}"]; cache = {}; F = None
    pur, chp, ga, gb, aal = (np.zeros((len(ORD), 10)) for _ in range(5))
    for i, o in enumerate(ORD):
        P = PL[f"{o}|raw"]; q, ycl, sp, cp, gA, gB, aA = {}, {}, {}, {}, {}, {}, {}; Qf = np.zeros((len(Y), 3)); yc = Y.copy()
        for s in S.sessions(m):
            r = np.flatnonzero(m["sess"] == s); mp = mapping(P[r], hard[r]); Qf[r] = soft_classes(soft[r], mp); yc[r] = mp[hard[r]]
            sp[s] = np.mean(yc[r] == Y[r]); cp[s] = np.mean(P[r].argmax(1) == Y[r]); gA[s] = fused_gain(o, s, lambda t: t, Qf)
            if arm in BARMS:
                F = F or CL.features(m); gl, al = [], []
                for n, trn, tst in S.pairs(m, s):
                    key = (s, n, tuple(mp))
                    if key not in cache:
                        cache[key] = CL.fit_predict(F, yc, trn, tst)["aE"]
                    al.append(S.cell_score(m, tst, cache[key])); gl.append(np.mean([S.cell_score(m, tst, S.fuse(c[tst], cache[key])) - S.cell_score(m, tst, c[tst]) for c in CH[o]]))
                gB[s], aA[s] = np.mean(gl), np.mean(al)
        pur[i], chp[i], ga[i] = by_person(sp), by_person(cp), by_person(gA)
        if arm in BARMS:
            gb[i], aal[i] = by_person(gB), by_person(aA)
        print(f"  {arm} {o}: purity {pur[i].mean():.3f} chain {chp[i].mean():.3f}", flush=True)
    return pur, chp, ga, gb, aal


BARMS = ("sess", "run")


def line(name, D):
    g = wmean(D, ORD); lo, hi = boot(D, ORD); return f"  {name:34s} {g.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}] {int((g > 0).sum()):2d}/10 | " + " ".join(f"{x:+.1f}" for x in g)


def causal(t0):
    """prefix form per t: purity of the mapped nearest-centroid labels on windows from t on, chain purity on the same windows, fused gain"""
    out = {t: [np.zeros((len(ORD), 10)) for _ in range(3)] for t in t0}
    for i, o in enumerate(ORD):
        key = f"{o.rstrip('0123456789')}|{o[len(o.rstrip('0123456789')):]}|all_loso"; perm = PERM[key]; P = PL[f"{o}|raw"]
        for t in t0:
            sp, cp, gg = {}, {}, {}; Qf = np.zeros((len(Y), 3)); pos = np.zeros(len(Y), bool)
            for s in S.sessions(m):
                d = perm[m["sess"][perm] == s]; pre, post = d[:t], d[t:]; assert set(d) == set(np.flatnonzero(m["sess"] == s)) and len(post) > 0
                lab, mu, _ = SP.cluster(CLU["Z"][pre], np.zeros(len(pre), int), np.random.default_rng(0)); mp = mapping(P[pre], lab)
                D = ((CLU["Z"][post][:, None] - mu[None]) ** 2).sum(-1); Dp = ((CLU["Z"][pre][:, None] - mu[None]) ** 2).sum(-1); sc = SP.EPS * np.median(Dp)
                Q = np.exp(-(D - D.min(1, keepdims=True)) / sc); Q /= Q.sum(1, keepdims=True); Qf[post] = soft_classes(Q, mp); pos[post] = True
                sp[s] = np.mean(mp[D.argmin(1)] == Y[post]); cp[s] = np.mean(P[post].argmax(1) == Y[post])
                gg[s] = fused_gain(o, s, lambda tt: tt[pos[tt]], Qf)
            for a, v in zip(out[t], (sp, cp, gg)):
                a[i] = by_person(v)
        print(f"  causal {o}: " + " ".join(f"t{t} purity {out[t][0][i].mean():.3f} chain {out[t][1][i].mean():.3f} gain {out[t][2][i].mean():+.2f}" for t in t0), flush=True)
    for t in t0:  # person x order cells with no scorable test window at all: filled with the person's mean over the other orders (counted)
        for a in out[t]:
            bad = np.isnan(a); print(f"  t={t}: {int(bad.sum())} empty person x order cells filled"); a[bad] = np.broadcast_to(np.nanmean(a, 0), a.shape)[bad]
    return out


def main():
    print(f"s259 transductive clustering; orders {len(ORD)}; chain posterior mapping; SMOKE={SMOKE}")
    print("diagnostic (labels, not used): best-permutation purity of the order-free clusters, per person:")
    for arm in BARMS:
        h = CLU[f"hard_{arm}"]; v = {s: max(np.mean(np.array(pm)[h[m["sess"] == s]] == Y[m["sess"] == s]) for pm in permutations(range(3))) for s in S.sessions(m)}
        print(f"  {arm}: mean {by_person(v).mean():.3f} | " + " ".join(f"{x:.2f}" for x in by_person(v)))
    for arm in BARMS:
        pur, chp, ga, gb, aal = noncausal(arm)
        print(f"\n=== non-causal, arm {arm} ({'PRIMARY, per-session balance' if arm == 'sess' else 'ORACLE, true run ids'})")
        print(f"  mapped-cluster purity {wmean(pur, ORD).mean():.3f} (per person " + " ".join(f"{x:.2f}" for x in wmean(pur, ORD)) + f") vs chain {wmean(chp, ORD).mean():.3f} (" + " ".join(f"{x:.2f}" for x in wmean(chp, ORD)) + ")")
        print("  gain over chain [80 % people x orders] people up | per person"); print(line("(a) chain x soft cluster", ga)); print(line("(b) chain x aE(cluster labels)", gb)); print(f"  (b) aE alone cell-BA {wmean(aal, ORD).mean():.2f} (chain on the same pairs not shown; see gains)")
    out = causal(TS)
    for t in TS:
        pur, chp, gg = out[t]; print(f"\n=== causal prefix t={t} (windows from t on; mapped nearest centroid; chain x soft cluster)")
        print(f"  purity {wmean(pur, ORD).mean():.3f} vs chain {wmean(chp, ORD).mean():.3f}"); print(line("(a) gain over chain", gg))
        for sc in ("within", "full", "runs"):
            ix = [k for k, o in enumerate(ORD) if o.startswith(sc)]
            if ix:
                print(f"    {sc:7s} purity {pur[ix].mean():.3f} chain {chp[ix].mean():.3f} gain {gg[ix].mean():+.2f}")
    print("\n(end-of-session t = the non-causal sess arm above)")


if __name__ == "__main__":
    main()
