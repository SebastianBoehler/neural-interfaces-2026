"""ORACLE (ledger s46a): v4.1 and v4.1 + own-space stage with perfect recognition of returning people.
Usage: python studies/12nov/oracle_gate.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
src = open(REPO + "/studies/12nov/v5_replay.py").read().split("\nres = []")[0]
exec(compile(src, "v5_replay_head", "exec"))  # route4 namespace ns, A, SC, staged(), REC, NM, O
real_bio = ns["bio_stream"]


def oracle_bio(proto, f, s, gi, names, t0):
    r, j, sc = np.zeros(len(gi)), np.full(len(gi), -1), np.full(len(gi), np.nan)
    hit = [i for i, q in enumerate(names) if q.endswith(O.PER[int(gi[0])])]
    if hit:
        r[:], j[:] = 1.0, hit[0]
    return r, j, sc


rows = []
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    ns["bio_stream"] = real_bio; REC.clear(); a0, _ = ns["system4"](bn, "max", 10); a1, _ = staged(a0)
    ns["bio_stream"] = oracle_bio; REC.clear(); b0, _ = ns["system4"](bn, "bio", 10); b1, _ = staged(b0)
    f = lambda x, y: SC.compare(x, y)  # noqa: E731
    r = [f(b0, a0), f(a1, a0), f(b1, a0)]; rows.append([[q["seen"], q["ds"][0], q["ds"][1], q["new"]] for q in r])
    print(f"{bn}: against v4.1 (real gate): oracle gate seen {r[0]['seen']:+.2f} (P04 {r[0]['ds'][0]:+.2f}, P10 {r[0]['ds'][1]:+.2f}) | stage, real gate {r[1]['seen']:+.2f} | "
          f"stage, oracle gate {r[2]['seen']:+.2f} (P04 {r[2]['ds'][0]:+.2f}, P10 {r[2]['ds'][1]:+.2f}) | new {r[0]['new']:+.3f}", flush=True)
m = np.mean(rows, 0)
print(f"mean of 6: oracle gate {m[0][0]:+.2f} (P04 {m[0][1]:+.2f}, P10 {m[0][2]:+.2f}); stage real gate {m[1][0]:+.2f}; stage oracle gate {m[2][0]:+.2f} (P04 {m[2][1]:+.2f}, P10 {m[2][2]:+.2f})")
