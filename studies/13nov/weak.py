"""WEAK (ledger s133, gap 3 of docs/SOLUTION_SPACE_MAP.md): fine-tune only the sessions that look weak without labels.
At window 30, where the fine-tuning starts, a session's confidence is the mean top probability of the shipped member
mix over its first 30 windows. FT (cum10) is applied where it is below a fixed quantile of the new-people sessions;
other sessions keep the frozen nets. Contexts: FT alone and FT+OCU (s130b), six bases, new people.
Usage: [BASES=B2_pool] [SMOKE=1] python studies/13nov/weak.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacklib import A, BASES, SLOT, SMOKE, Z, rng, run  # noqa: E402
p0 = "all_loso"; sess = Z[f"sess|{p0}"]; per = Z[f"person|{p0}"]; T0 = 30
QS = {"q50": 0.5, "q33": 1 / 3, "q67": 2 / 3, "inv50": 0.5}


def gates(bn):
    """per base: a session's confidence = mean top probability of this base's shipped mix over its first T0 windows;
    the threshold for a person's sessions is the quantile over the other nine people's sessions (review 13 Nov: no
    in-sample quantile, no B2_pool mix for every base)"""
    mix = A.mix(bn, p0, A.FAMS)[0]; ss = np.unique(sess); conf = {s: mix[np.flatnonzero(sess == s)[:T0]].max(1).mean() for s in ss}
    sp = {s: per[np.flatnonzero(sess == s)[0]] for s in ss}; out = {}
    for g, q in QS.items():
        keep = {s: (conf[s] >= np.quantile([conf[u] for u in ss if sp[u] != sp[s]], q)) if g == "inv50" else
                   (conf[s] < np.quantile([conf[u] for u in ss if sp[u] != sp[s]], q)) for s in ss}
        out[g] = np.array([keep[s] for s in sess])
        for a in SLOT:
            fr, cu = A.EXTRA[f"{a}_fr"][p0], A.EXTRA[f"{a}_ft"][p0]
            A.EXTRA[f"{a}_{g}"] = {p0: np.where(out[g][:, None], cu, fr), "seen_session": A.EXTRA[f"{a}_fr"]["seen_session"]}
    return out


ARMS = {"ungated": "ft", **{g: g for g in QS}}; CTX = {"FT": ("FT",), "FT+OCU": ("FT", "OCU")}
if SMOKE:
    ARMS = {k: ARMS[k] for k in ("ungated", "q50")}; BASES = BASES[:1]
R = {}
for bn in BASES:
    G_ = gates(bn); print(bn + " sessions fine-tuned: " + " ".join(f"{g} {len(np.unique(sess[m]))}" for g, m in G_.items()) + f" of {len(np.unique(sess))}; P04 in q50: {len(np.unique(sess[G_['q50'] & (per == 'P04')]))}", flush=True)
    R[bn, "frozen"] = run((), bn); R[bn, "OCU"] = run(("OCU",), bn)
    for c, S in CTX.items():
        for k, ft in ARMS.items():
            R[bn, c, k] = run(S, bn, ft=ft)
    print(bn + " " + " ".join(f"{c}:{k} {R[bn, c, k][0].mean() - R[bn, 'frozen' if c == 'FT' else 'OCU'][0].mean():+.2f}" for c in CTX for k in ARMS), flush=True)
for c in CTX:
    ref = "frozen" if c == "FT" else "OCU"; print(f"\n{c} context, against the same context without FT ({ref}), new people, mean over bases")
    for k in ARMS:
        dn = np.array([R[bn, c, k][0] - R[bn, ref][0] for bn in BASES]); ds = np.mean([(R[bn, c, k][1] - R[bn, ref][1]).mean() for bn in BASES]); d = dn.mean(0)
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
        print(f"  {k:8s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] bases +{(dn.mean(1) > 0).sum()}/{len(BASES)} people>=0 {(d >= 0).sum()}/10 P04 {d[3]:+.2f} | seen (A.system v3 path, not v4.1) {ds:+.2f} | " + " ".join(f"{x:+.1f}" for x in d), flush=True)
