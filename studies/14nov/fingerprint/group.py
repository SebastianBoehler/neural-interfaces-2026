"""FINGERPRINT ONLINE GROUPING (ledger s250). Per session, in delivery order, each window is assigned to a run group at
its arrival from windows <= t only (label fixed then), at most KMAX = 6 groups (public design: 2 acquisition + 4 online
runs), in the s250 metric fitted on the other people's true runs (metric.py). Forms:
  G1 sequential MAP, conjugate Gaussian mixture: within-run var f * v_t, run mean ~ N(m_t, (1 - f) * v_t), CRP alpha
  G2 online k-means with a new-group threshold (DP-means): new if min d^2 / (d * sbar_t) > th
  G3 nearest-earlier-window propagation: join the nearest earlier window's group; new if its d^2 / (d * sbar_t) > th
(v_t, m_t, sbar_t: the session's own causal spread, see online(); inner orders INNER, outer ORDERS)
Nested choice: metric setting and form parameters by mean online purity of the nine training people (each projected by a
metric fitted on the other eight) under the same orders. Scores the held-out person: metric 1-NN run identity, offline
Ward purity / ARI (k = true count), online purity / ARI / groups / false-merge rate (share of same-group window pairs from
different runs). Writes runs/fingerprint/groups_<form>.npz (key <order>: group label per replay row; RUN_IDS format).
Usage: [ORDERS=full0,full1,full2,full3] python group.py"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metric as M  # noqa: E402
C = M.C
KMAX, WARM = 6, 5
GRID = {"G1": list(itertools.product((0.1, 1.0, 10.0), (0.2, 0.35, 0.5))), "G2": [(t,) for t in (0.3, 0.5, 0.7, 1.0, 1.5)],
        "G3": [(t,) for t in (0.1, 0.2, 0.3, 0.5, 0.8)]}
ORDERS = os.environ.get("ORDERS", "full0,full1,full2,full3").split(","); INNER = os.environ.get("INNER", "full0,full1").split(",")


def online(Z, form, par, lam):
    """Scale-free in the session's own spread (the held-out person's scale differs up to 30x): v_t = per-direction variance
    of windows <= t, m_t their mean, sbar_t = 2 sum(v_t) / d (mean pairwise d^2 / d). No new group before t = WARM."""
    T, d = Z.shape; lab = np.zeros(T, int); S = np.zeros((KMAX, d)); n = np.zeros(KMAX); S[0] = Z[0]; n[0] = 1; k = 1
    cs, cq = Z[0].copy(), Z[0] ** 2
    for t in range(1, T):
        z = Z[t]; cs += z; cq += z ** 2; m = cs / (t + 1); v = np.maximum(cq / (t + 1) - m ** 2, 1e-9); sbar = 2 * v.sum() / d
        can = k < KMAX and t >= WARM
        if form == "G1":  # within-run variance f * v, run mean ~ N(m, (1 - f) * v), CRP alpha
            a, f = par; w, b = f * v, (1 - f) * v; vp = 1 / (1 / b + n[:k, None] / w); mp = vp * (m / b + S[:k] / w)
            ll = -0.5 * (((z - mp) ** 2) / (vp + w) + np.log(vp + w)).sum(1) + np.log(n[:k])
            lnew = -0.5 * (((z - m) ** 2) / (b + w) + np.log(b + w)).sum() + np.log(a)
            j = k if (can and lnew > ll.max()) else int(ll.argmax())
        elif form == "G2":
            d2 = ((S[:k] / n[:k, None] - z) ** 2).sum(1) / d / sbar; j = k if (can and d2.min() > par[0]) else int(d2.argmin())
        else:
            d2 = ((Z[:t] - z) ** 2).sum(1) / d / sbar; i = int(d2.argmin())
            if can and d2[i] > par[0]:
                j = k
            elif d2[i] <= par[0] or k < KMAX:
                j = lab[i]
            else:
                j = int((((S[:k] / n[:k, None] - z) ** 2).sum(1)).argmin())
        if j == k:
            k += 1
        lab[t] = j; S[j] += z; n[j] += 1
    return lab


def false_merge(lab, true):
    same = (lab[:, None] == lab[None]) & ~np.eye(len(lab), dtype=bool); return ((true[:, None] != true[None]) & same).sum() / max(same.sum(), 1)


def score_person(proj, lam, p, form, par, orders, keep=None):
    """mean online purity (and metrics) of person p's sessions over orders; keep: dict order -> label array to fill"""
    out = []
    for o in orders:
        for s in [s for s in C.SESSIONS if C.PER[np.flatnonzero(C.SES == s)[0]] == p]:
            rows = C.order_rows(o, s); lab = online(proj[rows], form, par, lam); tr = C.RUNI[rows]
            pur, ari = C.purity_ari(lab, tr); out.append((pur, ari, len(np.unique(lab)), false_merge(lab, tr)))
            if keep is not None:
                keep[o][rows] = lab
    return np.mean(out, 0)


def main():
    F = M.features(); res = {f: [] for f in GRID}; LAB = {f: {o: np.full(C.N, -1) for o in ORDERS} for f in GRID}; off = []
    print(f"FINGERPRINT GROUPING (s250): orders {ORDERS}, KMAX {KMAX}, {len(M.SETTINGS)} metric settings, nested choice")
    for h in C.PEOPLE:
        inner = {}  # (form, setting, par) -> list of inner purities
        for q in [q for q in C.PEOPLE if q != h]:
            for st in M.SETTINGS:
                proj, lam = M.fit(F, (C.PER != h) & (C.PER != q), st); Pq = np.zeros((C.N, st[1])); m = C.PER == q; Pq[m] = proj(F[m])
                for form, pars in GRID.items():
                    for par in pars:
                        inner.setdefault((form, st, par), []).append(score_person(Pq, lam, q, form, par, INNER)[0])
        hm = C.PER == h
        for form in GRID:
            (fm, st, par), v = max(((k, v) for k, v in inner.items() if k[0] == form), key=lambda kv: np.mean(kv[1]))
            proj, lam = M.fit(F, ~hm, st); Ph = np.zeros((C.N, st[1])); Ph[hm] = proj(F[hm])
            r = score_person(Ph, lam, h, form, par, ORDERS, keep=LAB[form]); res[form].append(r)
            print(f"  {h} {form} setting {st} par {par} (inner purity {np.mean(v):.3f}) | held-out purity {r[0]:.3f} ARI {r[1]:.3f} groups {r[2]:.1f} false-merge {r[3]:.3f}", flush=True)
            if form == "G1":
                from sklearn.cluster import AgglomerativeClustering
                for s in [s for s in C.SESSIONS if C.PER[np.flatnonzero(C.SES == s)[0]] == h]:
                    rows = np.flatnonzero(C.SES == s); Zs = Ph[rows]; tr = C.RUNI[rows]; k = len(np.unique(tr))
                    cw = AgglomerativeClustering(k, linkage="ward").fit_predict(Zs); off.append((M.nn1(Zs, tr), *C.purity_ari(cw, tr)))
    o = np.mean(off, 0)
    print(f"OFFLINE (metric of the nested G1 setting, per session): 1-NN run identity {o[0]:.3f} | Ward purity {o[1]:.3f} ARI {o[2]:.3f}")
    print("ONLINE (held-out people, mean over people; nested choice): purity [80 % person interval] | ARI | groups/session | false-merge")
    rng = np.random.default_rng(0)
    for form, r in res.items():
        r = np.array(r); bt = np.percentile([r[rng.integers(0, len(r), len(r)), 0].mean() for _ in range(4000)], [10, 90])
        print(f"  {form}: purity {r[:, 0].mean():.3f} [{bt[0]:.3f}, {bt[1]:.3f}] | ARI {r[:, 1].mean():.3f} | groups {r[:, 2].mean():.2f} | false-merge {r[:, 3].mean():.3f}")
        assert all((v[C.PER != "x"] >= 0).all() for v in LAB[form].values()), "a row got no group"
        np.savez(f"{C.REPO}/runs/fingerprint/groups_{form}.npz", **{k: v.astype(str) for k, v in LAB[form].items()})


if __name__ == "__main__":
    main()
