"""s277 form 2 collect: person-equal BA of probe A/B and FT A/B/0f (mean of 3 seeds) over ten folds; FT A/B minus 0f, minus supervised arm 3 (form 1, evidence/ssl277_15nov).
Usage: collect2.py [runs/probe2_ssl]  (80 % interval resamples the ten people, 2000 draws)"""
import glob, json, sys
import numpy as np
fs = sorted(glob.glob(f"{sys.argv[1] if len(sys.argv) > 1 else 'runs/probe2'}/fold[0-9].json")); R = [json.load(open(f)) for f in fs]
assert len(R) == 10 and [r["fold"] for r in R] == list(range(10)), len(R)
S = [json.load(open(f"evidence/ssl277_15nov/fold{i}.json")) for i in range(10)]
for r, s in zip(R, S): assert r["held"] == s["held"] and r["pre_steps"] == 30000 and r["ft_steps"] == 6000 and r["pretrained_differ"] and set(r["ft"]) == {f"{a}_s{k}" for a in ("A", "B", "0f") for k in range(3)}
who = [r["held"].split("/")[-1] for r in R]; rng = np.random.default_rng(0); idx = rng.integers(0, 10, (2000, 10))
v = {"probeA": [r["probe"]["A"]["ba"] for r in R], "probeB": [r["probe"]["B"]["ba"] for r in R], "sup3": [s["arms"]["3"]["ba"] for s in S], "arm0probe": [s["arms"]["0"]["ba"] for s in S]}
for a in ("A", "B", "0f"): v["FT" + a] = [np.mean([r["ft"][f"{a}_s{k}"]["ba"] for k in range(3)]) for r in R]
v = {k: np.array(x) * 100 for k, x in v.items()}
def rep(n, d):
    b = d[idx].mean(1); lo, hi = np.percentile(b, [10, 90]); p = lambda w: f"{d[who.index(w)]:+.1f}"
    print(f"{n:28s} {d.mean():+6.2f} [{lo:+.2f},{hi:+.2f}] up {(d > 0).sum()}/10 P04 {p('P04')} P10 {p('P10')}")
for k, x in v.items(): print(f"{k:10s} BA {x.mean():.2f}")
for a, b in (("probeA", "arm0probe"), ("probeB", "arm0probe"), ("FTA", "FT0f"), ("FTB", "FT0f"), ("FTA", "sup3"), ("FTB", "sup3"), ("FT0f", "sup3")): rep(f"{a} - {b}", v[a] - v[b])
print("per seed FT mean:", {a: [round(float(np.mean([r["ft"][f"{a}_s{k}"]["ba"] for r in R]) * 100), 2) for k in range(3)] for a in ("A", "B", "0f")})
print("per person FTA/FTB/FT0f/sup3:", {w: [round(float(v[k][i]), 1) for k in ("FTA", "FTB", "FT0f", "sup3")] for i, w in enumerate(who)})
