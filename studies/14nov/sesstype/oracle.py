"""s236: per-session oracle ceiling and nested descriptor rules for v6f's eye-view weight and stage constants.
Per-window outputs (recording layout, '<base>|V6Fr' -> (2655, 3)) per order and setting:
  shipped  runs/chainout/<order>.npz (equal to s231_<order>_w1.npz)
  eye      runs/chainout/s231_<order>_w0.npz (eye view off); s236_<order>_eyex2 / eyex3 (one / two extra copies; EYEW
           scaling is a no-op for w > 0 because SessionEM z-scores its columns)
  stages   runs/chainout/s236_<order>_<knob><v>.npz (studies/14nov/sesstype/launch.sh)
Scores: session BA = mean of its cells' BA (cell = session x interface); person score = mean of its cells; mean over the
six bases. Causal emission for rules: windows at delivery positions < K keep the shipped output, from K on the chosen
setting's own output. Descriptors from runs/sesstype/desc.npz (desc.py). Splits: all 252 ordered 5/5 person splits.
Usage: [K=40] [ALLOW_MISSING=1] [SETS=...] python studies/14nov/sesstype/oracle.py"""
import itertools, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from desc import delivery_pos  # noqa: E402
B = np.load(paths.DATA + "/bases.npz", allow_pickle=True)
ORDERS = os.environ.get("ORDERS", "within0,within1,full0,full1,runs0,runs1").split(",")
SCHEMES = list(dict.fromkeys(o.rstrip("0123456789") for o in ORDERS))
BASES = ["B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"]
EYE = ["w0", "eyex2", "eyex3"]; STG = ["em_lam0.25", "em_lam1.0", "ts_lam0.075", "ts_lam0.3", "bts2", "bts8"]
SETS = os.environ.get("SETS", ",".join(EYE + STG)).split(",")
K = int(os.environ.get("K", 40)); ARM = os.environ.get("ARM", "V6Fr")
y, sess, cell, person = (B[f"{k}|all_loso"] for k in ("y", "sess", "cell", "person")); y = y.astype(int)
SESS = np.unique(sess); PEOPLE = sorted(set(person)); SP = np.array([person[sess == s][0] for s in SESS])
NC = {s: len(np.unique(cell[sess == s])) for s in SESS}; assert len(set(NC.values())) == 1, "unequal cells per session"


def path(o, st):
    return f"{REPO}/runs/chainout/{o}.npz" if st == "ship" else f"{REPO}/runs/chainout/s231_{o}_{st}.npz" if st == "w0" else \
        f"{REPO}/runs/chainout/s236_{o}_{st}.npz"


def sess_ba(P):
    """(14,) session BA (mean of the session's cells' BA) for one output array."""
    pr = P.argmax(1); out = np.zeros(len(SESS))
    for i, s in enumerate(SESS):
        m = sess == s
        out[i] = np.mean([100 * np.mean([np.mean(pr[m & (cell == c) & (y == k)] == k) for k in np.unique(y[m & (cell == c)])])
                          for c in np.unique(cell[m])])
    return out


def load():
    """-> settings used, full[o] (S+1, 14) base-mean session BA, causal[o] (S+1, 14) with the shipped output before K."""
    have = [st for st in SETS if all(os.path.exists(path(o, st)) for o in ORDERS)]; miss = sorted(set(SETS) - set(have))
    assert not miss or os.environ.get("ALLOW_MISSING"), f"missing settings files: {miss}"
    if miss:
        print("MISSING (left out):", miss, flush=True)
    full, causal = {}, {}
    for o in ORDERS:
        pos = delivery_pos(o); zs = np.load(path(o, "ship")); F, C = [], []
        ref = {bn: zs[f"{bn}|{ARM}"] for bn in BASES}
        for st in ["ship"] + have:
            z = np.load(path(o, st)) if st != "ship" else zs
            assert st not in EYE or np.abs(z["B2_pool|P0r"] - zs["B2_pool|P0r"]).max() < 1e-9, f"{o} {st}: P0r differs (EYEW must not touch it)"
            assert st == "ship" or np.abs(z[f"B2_pool|{ARM}"] - ref["B2_pool"]).max() > 0, f"{o} {st}: identical to shipped (silent no-op)"
            F.append(np.mean([sess_ba(z[f"{bn}|{ARM}"]) for bn in BASES], 0))
            C.append(np.mean([sess_ba(np.where((pos < K)[:, None], ref[bn], z[f"{bn}|{ARM}"])) for bn in BASES], 0))
        full[o], causal[o] = np.array(F), np.array(C)
        assert np.allclose(F[0], C[0])
    return have, full, causal


def pgain(dS, people):
    """person gain = mean over the person's sessions (equal cells per session) of the session change."""
    return np.array([dS[SP == p].mean() for p in people])


def band(v, rng=np.random.default_rng(0)):
    bt = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)]; return np.percentile(bt, [10, 90])


def report_oracle(have, full):
    groups = {"all": list(range(1, len(have) + 1)), "EYEW": [i + 1 for i, s in enumerate(have) if s in EYE],
              "stages": [i + 1 for i, s in enumerate(have) if s in STG]}
    print(f"\nORACLE CEILING (per session best setting, labels; new people, person score change vs shipped, mean over bases)")
    for st in have:  # global (every session) change of each setting, for reference
        g = np.mean([pgain(full[o][1 + have.index(st)] - full[o][0], PEOPLE) for o in ORDERS], 0)
        print(f"  always {st:12s} {g.mean():+.2f} [{band(g)[0]:+.2f}, {band(g)[1]:+.2f}]")
    for gname, idx in groups.items():
        if not idx:
            continue
        line = []
        for sc in SCHEMES + ["all"]:
            os_ = [o for o in ORDERS if sc == "all" or o.rstrip("0123456789") == sc]
            g = np.mean([pgain(np.max(full[o][[0] + idx], 0) - full[o][0], PEOPLE) for o in os_], 0)
            line.append(f"{sc} {g.mean():+.2f} [{band(g)[0]:+.2f}, {band(g)[1]:+.2f}]")
        print(f"  oracle {gname:7s} " + " | ".join(line))
    pick = {o: np.argmax(full[o], 0) for o in ORDERS}
    print("  oracle picks (all settings) per session, by order:")
    for i, s in enumerate(SESS):
        print(f"    {SP[i]} s{s}: " + " ".join((["ship"] + have)[pick[o][i]] for o in ORDERS))


def rules(have, causal, desc, dnames, idx_pool):
    """candidate rules: (d, s, dir, theta) and 'always s'. Returns a function scoring a candidate on (orders, people)."""
    return [(None, si, 0, 0.0) for si in idx_pool] + [(d, si, sg, None) for d in dnames for si in idx_pool for sg in (1, -1)]


def gains_of(cand, causal, desc, orders, people, theta=None):
    d, si, sg, th = cand; th = th if theta is None else theta
    g = []
    for o in orders:
        dS = causal[o][si] - causal[o][0]
        if d is not None:
            dS = np.where(sg * (desc[o][:, d] - th) > 0, dS, 0.0)
        g.append(pgain(dS, people))
    return np.mean(g, 0)


def fit(cands, causal, desc, train):
    """best candidate on the training people (all orders pooled); thresholds = midpoints of the training sessions' values.
    Vectorised: gain = mean over orders of sum_i w_i dS[o, i] [rule fires on session i], w_i = 1 / (n sessions of the
    person x n training people) on training sessions, 0 elsewhere (equal to gains_of(...).mean())."""
    best, bg = (None, 0, 0, 0.0), 0.0  # shipped: gain 0; a candidate must beat it strictly
    tr = np.isin(SP, train); w = np.where(tr, 1.0 / np.array([np.sum(SP == p) for p in SP]) / len(train), 0.0)
    for d, si, sg, _ in cands:
        dS = np.array([causal[o][si] - causal[o][0] for o in ORDERS])  # (O, 14)
        if d is None:
            g = (dS * w).sum(1).mean()
            if g > bg + 1e-12:
                best, bg = (None, si, 0, 0.0), g
            continue
        D = np.array([desc[o][:, d] for o in ORDERS]); v = np.unique(D[:, tr]); ths = (v[1:] + v[:-1]) / 2
        if not len(ths):
            continue
        fire = sg * (D[None] - ths[:, None, None]) > 0  # (T, O, 14)
        g = (fire * (dS * w)[None]).sum(2).mean(1); j = int(np.argmax(g))
        if g[j] > bg + 1e-12:
            best, bg = (d, si, sg, ths[j]), g[j]
    assert best[1] == 0 or abs(gains_of(best, causal, None if best[0] is None else desc, ORDERS, train).mean() - bg) < 1e-9, "fit != gains_of"
    return best


def nested(have, causal, desc, dsel, label):
    idx = list(range(1, len(have) + 1)); cands = rules(have, causal, desc, dsel, idx)
    if not dsel:
        cands = [c for c in cands if c[0] is None]
    split_g = {sc: [] for sc in SCHEMES + ["all"]}; per = {p: {sc: [] for sc in SCHEMES + ["all"]} for p in PEOPLE}; chosen = {}
    for tr in itertools.combinations(PEOPLE, 5):
        te = [p for p in PEOPLE if p not in tr]
        for train, test in ((list(tr), te), (te, list(tr))):
            c = fit(cands, causal, desc, train); key = ("ship" if c[0] is None and c[1] == 0 else
                                                       (f"always {have[c[1] - 1]}" if c[0] is None else f"{DN[c[0]]}{'>' if c[2] > 0 else '<'} -> {have[c[1] - 1]}"))
            chosen[key] = chosen.get(key, 0) + 1
            for sc in SCHEMES + ["all"]:
                os_ = [o for o in ORDERS if sc == "all" or o.rstrip("0123456789") == sc]
                g = np.zeros(5) if c[0] is None and c[1] == 0 else gains_of(c, causal, desc, os_, test)
                split_g[sc].append(g.mean())
                for p, v in zip(test, g):
                    per[p][sc].append(v)
    out = []
    for sc in SCHEMES + ["all"]:
        sg = np.array(split_g[sc]); pp = np.array([np.mean(per[p][sc]) for p in PEOPLE]); b = band(pp)
        out.append(f"{sc} {sg.mean():+.2f} (splits 10-90 % {np.percentile(sg, 10):+.2f}..{np.percentile(sg, 90):+.2f}; person band [{b[0]:+.2f}, {b[1]:+.2f}])")
    print(f"  {label:18s} " + " | ".join(out))
    top = sorted(chosen.items(), key=lambda kv: -kv[1])[:4]
    print("      chosen: " + ", ".join(f"{k} {v}" for k, v in top))
    if label.startswith("FULLY NESTED"):
        pp = np.array([np.mean(per[p]["all"]) for p in PEOPLE])
        print("      per person (all orders): " + " ".join(f"{p} {v:+.2f}" for p, v in zip(PEOPLE, pp)) + f" | without P10 {np.delete(pp, PEOPLE.index('P10')).mean():+.2f}")


DN = ["e1", "e2", "a", "s"]


def main():
    have, full, causal = load()
    report_oracle(have, full)
    dz = np.load(REPO + "/runs/sesstype/desc.npz"); assert list(dz["names"]) == DN and (dz["sess"] == SESS).all()
    desc = {o: dz[f"{o}|K{K}"] for o in ORDERS}
    print(f"\nNESTED RULES (K {K}: descriptors from the first {K} delivered windows, shipped output before K; held-out gain, new people)")
    c_or = np.mean([[pgain(np.max(causal[o], 0) - causal[o][0], PEOPLE).mean()] for o in ORDERS])
    print(f"  oracle with the causal switch at K: {c_or:+.2f}")
    nested(have, causal, desc, list(range(4)), "FULLY NESTED")
    nested(have, causal, desc, [], "global only")
    for d in range(4):
        nested(have, causal, desc, [d], f"descriptor {DN[d]}")


if __name__ == "__main__":
    main()
