"""TWOSYS (ledger s68): two complete chains on disjoint member families, fused at the end. Usage: python studies/12nov/twosys.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; p = "all_loso"; CORES = ["core1", "hilb"]; NETS = ["tf", "shallow", "sbnfb", "sbnsinc", "sbnfbl"]; rng = np.random.default_rng(0)
pp = lambda P: 100 * np.asarray(SC.person(P, p)) if np.asarray(SC.person(P, p)).max() <= 1 else np.asarray(SC.person(P, p))  # noqa: E731


def boot(d):
    b = [d[rng.integers(0, 10, 10)].mean() for _ in range(4000)]; return f"{d.mean():+.2f} [{np.percentile(b, 10):+.2f}, {np.percentile(b, 90):+.2f}]"


def fuse(Pa, Pb, wa):
    L = wa * np.log(np.clip(Pa, 1e-9, 1)) + (1 - wa) * np.log(np.clip(Pb, 1e-9, 1)); e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


for bn in env.EXPL:
    full = A.system(bn, A.FAMS, routing=False)["out"][p]; ref = pp(full); print(f"{bn}: single chain (no routing) new {ref.mean():.2f} | P04 {ref[3]:.1f}", flush=True)
    for tag, fa, fb in (("teacher in both", CORES + ["teacher"], NETS + ["teacher"]), ("teacher with the cores", CORES + ["teacher"], NETS), ("teacher with the nets", CORES, NETS + ["teacher"])):
        Pa = A.system(bn, fa, routing=False)["out"][p]; Pb = A.system(bn, fb, routing=False)["out"][p]; a, b = pp(Pa), pp(Pb)
        print(f"  {tag}: chain A {a.mean():.2f} (P04 {a[3]:.1f}) | chain B {b.mean():.2f} (P04 {b[3]:.1f}) | agreement of labels {100 * np.mean(Pa.argmax(1) == Pb.argmax(1)):.1f}%")
        for wa in (0.3, 0.5, 0.7):
            d = pp(fuse(Pa, Pb, wa)) - ref; print(f"     fused, A exponent {wa}: new {boot(d)} | P04 {d[3]:+.1f} | people >= 0: {(d >= 0).sum()}/10")
        mixf = 0.5 * Pa + 0.5 * Pb; d = pp(mixf) - ref; print(f"     averaged: new {boot(d)} | P04 {d[3]:+.1f}", flush=True)
