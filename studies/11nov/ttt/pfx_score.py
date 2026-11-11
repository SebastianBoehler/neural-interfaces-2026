"""Score the PFT proxy (ledger s32): per person, standalone balanced accuracy of personally fine-tuned nets on runs 4-6
of session 1 against the same nets frozen (mean over families and seeds). Usage: python pfx_score.py PROBS_PFX"""
import glob, os, sys
import numpy as np
from scipy.stats import spearmanr
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/experiments")
import protocol
d = protocol.load(); D = sys.argv[1]; per = np.array([s[-3:] for s in d["subject"]]); y = d["y"]
files = sorted(glob.glob(f"{D}/pfx_*_base-proxy-s*.npz")); people = sorted(set(per[d["split"] != "external"]))
VARS = ["base", "pf20", "pf60", "pfh60"]; arch_of = lambda f: os.path.basename(f).split("_")[1]  # noqa: E731


def ba(P, idx, q):
    m = per[idx] == q; return 100 * np.mean([np.mean(P[m & (y[idx] == k)].argmax(1) == k) for k in range(3)])


R = {v: [] for v in VARS}; tags = []
for f in files:
    zs = {v: np.load(f.replace("_base-", f"_{v}-")) for v in VARS if os.path.exists(f.replace("_base-", f"_{v}-"))}
    if len(zs) < len(VARS):
        continue
    tags.append(os.path.basename(f)[4:].replace("_base-proxy-", " "))
    for v in VARS:
        R[v].append([ba(zs[v]["proba"], zs[v]["index"], q) for q in people])
R = {v: np.array(a) for v, a in R.items()}; print(f"{len(tags)} nets: {tags}")
print("person  frozen | gain pf20  pf60  pfh60")
for i, q in enumerate(people):
    print(f"{q}     {R['base'][:, i].mean():5.1f} | " + "  ".join(f"{(R[v][:, i] - R['base'][:, i]).mean():+5.1f}" for v in VARS[1:]))
for v in VARS[1:]:
    g = (R[v] - R["base"]).mean(0); i4 = people.index("P04")
    print(f"{v}: mean gain {g.mean():+.2f}; people with gain >= half of P04's ({g[i4]:+.1f}): {int((g >= g[i4] / 2).sum())}/10; people >= 0: {int((g >= 0).sum())}/10; "
          f"Spearman(gain, frozen weakness) {spearmanr(g, 100 - R['base'].mean(0)).statistic:+.2f}")
for a in sorted({t.split(" ")[0] for t in tags}):
    k = [i for i, t in enumerate(tags) if t.split(" ")[0] == a]
    print(f"  {a:5s} ({len(k)} nets): frozen {R['base'][k].mean():.1f}, pf60 {(R['pf60'][k] - R['base'][k]).mean():+.2f}, pfh60 {(R['pfh60'][k] - R['base'][k]).mean():+.2f}")
