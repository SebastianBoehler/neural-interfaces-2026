"""s290 part 1 collect: per arm, ten-person mean of lr10 vs the flat arm P1c (same code, asserted equal to s263 Z lr10 = 69.29), delta, people up.
Usage: collect.py [DIR=runs/s290]; run in the Galvani work dir ni26_s290 (or after rsync). Rule: >= +0.5 and >= 7/10 up -> TSEM/chain screen; -0.5..+0.5 undecided; < -0.5 closed in that form."""
import glob, json, sys
D = sys.argv[1] if len(sys.argv) > 1 else "runs/s290"
ARMS = ["P1a", "P1b", "P1c", "P2a", "P2b", "P2c", "P3a", "P3b"]
R = {a: {j["person"]: j["ba"]["lr10"] for j in map(lambda f: json.load(open(f)), glob.glob(f"{D}/{a}_fold*.json"))} for a in ARMS}
b = R["P1c"]
print("folds per arm", {a: len(v) for a, v in R.items()})
print(f"flat P1c mean {100 * sum(b.values()) / max(len(b), 1):.2f} (reference 69.29 when n=10)")
for a in ARMS[:2] + ARMS[3:]:
    ps = sorted(set(b) & set(R[a]))
    if ps:
        dl = [100 * (R[a][p] - b[p]) for p in ps]
        print(f"{a} n={len(ps)} mean {100 * sum(R[a][p] for p in ps) / len(ps):.2f} delta {sum(dl) / len(dl):+.2f} up {sum(x > 0 for x in dl)}/{len(ps)}  P04 {dl[ps.index('P04')] if 'P04' in ps else float('nan'):+.2f} P10 {dl[ps.index('P10')] if 'P10' in ps else float('nan'):+.2f}")
