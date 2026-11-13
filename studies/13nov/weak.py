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
mix = A.mix("B2_pool", p0, A.FAMS)[0]; conf = {}
for s in np.unique(sess):
    r = np.flatnonzero(sess == s); conf[s] = mix[r[:T0], :].max(1).mean()
cv = np.array(list(conf.values())); rowc = np.array([conf[s] for s in sess])
GATES = {"q50": rowc < np.quantile(cv, 0.5), "q33": rowc < np.quantile(cv, 1 / 3), "q67": rowc < np.quantile(cv, 2 / 3), "inv50": rowc >= np.quantile(cv, 0.5)}
for g, m in GATES.items():
    for a in SLOT:
        fr, cu = A.EXTRA[f"{a}_fr"][p0], A.EXTRA[f"{a}_ft"][p0]
        A.EXTRA[f"{a}_{g}"] = {p0: np.where(m[:, None], cu, fr), "seen_session": A.EXTRA[f"{a}_fr"]["seen_session"]}
    print(f"gate {g}: {len(np.unique(sess[m]))} of {len(cv)} sessions fine-tuned; P04 sessions in it {len(np.unique(sess[m & (per == 'P04')]))} of {len(np.unique(sess[per == 'P04']))}", flush=True)
ARMS = {"ungated": "ft", **{g: g for g in GATES}}; CTX = {"FT": ("FT",), "FT+OCU": ("FT", "OCU")}
if SMOKE:
    ARMS = {k: ARMS[k] for k in ("ungated", "q50")}; BASES = BASES[:1]
R = {}
for bn in BASES:
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
        print(f"  {k:8s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] bases +{(dn.mean(1) > 0).sum()}/{len(BASES)} people>=0 {(d >= 0).sum()}/10 P04 {d[3]:+.2f} | seen {ds:+.2f} | " + " ".join(f"{x:+.1f}" for x in d), flush=True)
