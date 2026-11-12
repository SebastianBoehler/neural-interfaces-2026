"""UPW (ledger s50), part 2: the up-weighted personal core in the v4.1 chain replay, seen sessions, with and without the
own-space stage. New-person rows are untouched (the file is NaN there). Usage: python studies/12nov/upw_chain.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
exec(compile(open(REPO + "/studies/12nov/v5_replay.py").read().split("\nres = []")[0], "v5_replay_head", "exec"))  # ns, A, SC, staged, REC, O
UP = np.load(REPO + "/runs/upw_core.npz"); idx = A.pl.B["idx|seen_session"].astype(int); C0 = A.C1["seen_session"].copy(); _members = A.members
d1 = np.abs(UP["P|1"][idx] - C0).max(); print(f"harness: w = 1 against the stored core1 posteriors on the seen rows: max abs difference {d1:.2e}", flush=True)
assert d1 < 1e-4, "the refit does not reproduce the stored core1; no result is read"
CUR = [None]


def members(bn, p):
    mem = _members(bn, p)  # the Hilbert slot is derived from the original core1 inside
    if p == "seen_session" and CUR[0] is not None:
        mem = dict(mem); mem["c1"] = CUR[0]
    return mem


A.members = members
for bn in ("B2_pool", "B2_s345"):
    CUR[0] = None; REC.clear(); base, _ = ns["system4"](bn, "max", 10); bst, _ = staged(base); r0 = SC.compare(bst, base)
    print(f"{bn}: stage alone seen {r0['seen']:+.2f} (P04 {r0['ds'][0]:+.2f}, P10 {r0['ds'][1]:+.2f})", flush=True)
    for w in (3, 5, 10, 20):
        P = UP[f"P|{w}"][idx]; alone = [100 * np.mean(P[A.SC.B["person|seen_session"] == q].argmax(1) == A.SC.B["y|seen_session"][A.SC.B["person|seen_session"] == q]) for q in ("P04", "P10")]
        CUR[0] = P; REC.clear(); out, _ = ns["system4"](bn, "max", 10); ost, _ = staged(out); r, rs = SC.compare(out, base), SC.compare(ost, base)
        print(f"   w {w:2d}: core alone P04 {alone[0]:.1f} P10 {alone[1]:.1f} | chain seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f}) | with stage {rs['seen']:+.2f} "
              f"(P04 {rs['ds'][0]:+.2f}, P10 {rs['ds'][1]:+.2f}) | new {r['new']:+.3f}", flush=True)
CUR[0] = None
