"""s278 step 3 collect: python collect.py runs/shin278/S16 [out.txt]. Guards: 10 folds, one cache key, 0 Shin rows in a0, a1w0 equals a0 (probabilities)."""
import json, sys
import numpy as np
D = sys.argv[1]
fs = [json.load(open(f"{D}/fold{i}.json")) for i in range(10)]
assert len({f["key"] for f in fs}) == 1, "mixed cache keys"
P = [f["person"] for f in fs]; arms = list(fs[0]["ba"])
assert all(f["n_train"]["a0"][1] == 0 and f["arms"]["a0"]["dr"] is False for f in fs)
L = [f"w0 guard: max |p(a1w0) - p(a0)| over folds {max(f['w0_maxdiff'] for f in fs):.2e}"]
b = {a: np.array([f["ba"][a] for f in fs]) for a in arms}; r = {a: np.array([f["word_recall"][a] for f in fs]) for a in arms}
L.append("people " + " ".join(P))
for a in arms:
    d = b[a] - b["a0"]; i4, i10 = P.index("P04"), P.index("P10")
    L.append(f"{a:7s} BA {100 * b[a].mean():.2f} vs a0 {100 * d.mean():+.2f} up {(d > 0).sum()}/10 P04 {100 * d[i4]:+.1f} P10 {100 * d[i10]:+.1f} | WORD recall {100 * r[a].mean():.1f} ({100 * (r[a] - r['a0']).mean():+.1f}) | per person " + " ".join(f"{100 * x:+.1f}" for x in d))
d = b["a2w0.3"] - b["a0"]; u = int((d > 0).sum()); m = d.mean()
L.append(f"rule on arm2 w0.3 (pred +0.3; arm1 w0.3 pred -1.5 -> {100 * (b['a1w0.3'] - b['a0']).mean():+.2f}): delta {100 * m:+.2f}, up {u}/10 -> "
         + ("pre-register net form and chain check" if m >= .01 and u >= 7 else "closed in this form" if m < -.005 else "undecided"))
print("\n".join(L))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write("\n".join(L) + "\n")
