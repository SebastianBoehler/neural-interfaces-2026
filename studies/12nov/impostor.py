"""IMPOSTOR (ledger s65): every new-person session forced to route to its most similar public person.
Usage: python studies/12nov/impostor.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
exec(compile(open(REPO + "/studies/12nov/v5_replay.py").read().split("\nres = []")[0], "v5_replay_head", "exec"))  # ns, A, R3, SC, staged, REC, O
real_bio = ns["bio_stream"]


def forced(proto, f, s, gi, names, t0):
    r, j, sc = real_bio(proto, f, s, gi, names, t0)
    if proto != "all_loso":
        return r, j, sc
    top = np.bincount(j[j >= 0].astype(int), minlength=len(names)).argmax() if (j >= 0).any() else 0  # the person the gate prefers most often
    r2, j2 = np.zeros(len(gi)), np.full(len(gi), top); r2[t0:] = 1.0; return r2, j2, sc


for bn in ("B2_pool", "B2_s345"):
    ns["bio_stream"] = real_bio; REC.clear(); base, _ = ns["system4"](bn, "max", 10)
    ns["bio_stream"] = forced; REC.clear(); f4, info = ns["system4"](bn, "bio", 10); f5, touched = staged(f4)
    r4, r5 = SC.compare(f4, base), SC.compare(f5, base); n = sum(1 for t in touched if t[0] == "all_loso")
    print(f"{bn}: {n} new-person sessions forced | v4.1 new {r4['new']:+.2f} | v5.1 new {r5['new']:+.2f} | extra loss from the stage {r5['new'] - r4['new']:+.2f}")
    print("   per person v4.1: " + " ".join(f"{q[1:]}:{v:+.1f}" for q, v in zip(SC.PEOPLE, r4["dn"])) + "\n   per person v5.1: " + " ".join(f"{q[1:]}:{v:+.1f}" for q, v in zip(SC.PEOPLE, r5["dn"])), flush=True)
    if bn == "B2_pool":
        print("   routed to: " + " ".join(f"{t[1]}->{t[2]}" for t in touched if t[0] == "all_loso"))
ns["bio_stream"] = real_bio
