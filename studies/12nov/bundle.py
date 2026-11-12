"""BUNDLE (ledger s57b): personal TSEM template prior (kappa) and slot gain on top of the own-space stage, v4.1 replay.
Usage: [BASES=..] python studies/12nov/bundle.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
exec(compile(open(REPO + "/studies/12nov/v5_replay.py").read().split("\nres = []")[0], "v5_replay_head", "exec"))  # ns, A, R3, SC, staged, REC
for bn in os.environ.get("BASES", "B2_pool,B2_s345").split(","):
    ns["KAPPA"], R3.G3 = 0.0, 3.0; REC.clear(); base, _ = ns["system4"](bn, "max", 10); st0, _ = staged(base); r0 = SC.compare(st0, base)
    print(f"{bn}: stage alone seen {r0['seen']:+.2f} (P04 {r0['ds'][0]:+.2f}, P10 {r0['ds'][1]:+.2f})", flush=True)
    for kap in (0.0, 10.0, 20.0):
        for g3 in (3.0, 4.5):
            if kap == 0 and g3 == 3.0:
                continue
            ns["KAPPA"], R3.G3 = kap, g3; REC.clear(); o, _ = ns["system4"](bn, "max", 10); os_, _ = staged(o); r, rs = SC.compare(o, base), SC.compare(os_, base)
            print(f"   kappa {kap:4.0f} gain {g3}: without stage {r['seen']:+.2f} | with stage {rs['seen']:+.2f} (P04 {rs['ds'][0]:+.2f}, P10 {rs['ds'][1]:+.2f}) "
                  f"= {rs['seen'] - r0['seen']:+.2f} over the stage alone | new {rs['new']:+.3f}", flush=True)
ns["KAPPA"], R3.G3 = 0.0, 3.0
