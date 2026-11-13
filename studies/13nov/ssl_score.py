"""Score SSL1 (ledger s147): the FBLight SessionBN family (slot 8) replaced by the self-supervised-trunk nets against
their fresh-trunk twins, new people, six bases (A.system; new people only, seen keeps the shipped slot 8 for both arms).
Usage: [SEEDS=0-11] python studies/13nov/ssl_score.py PROBS_DIR"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; D = sys.argv[1]; rng = np.random.default_rng(0); IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}
_a, _b = (int(x) for x in os.environ.get("SEEDS", "0-11").split("-")); SEEDS = range(_a, _b + 1)
print(f"SSL1 scorer: {D}, seeds {list(SEEDS)}, slot 8 (sbnfbl) swapped, A.system (routing v3 dev chain; new people only); REFERENCE: a fresh-trunk study twin, NOT v5.1b as uploaded", flush=True)
for arm in ("twin", "ssl"):
    zs = [np.load(f"{D}/ssl_fbl_{arm}-all_loso-s{s}.npz") for s in SEEDS]; assert all((z["index"] == IDX["all_loso"]).all() for z in zs), arm
    A.EXTRA[f"fbl_{arm}"] = {"all_loso": np.mean([z["proba"] for z in zs], 0), "seen_session": A.members("B2_pool", "seen_session")[8]}
    print(f"{arm} alone, new: {SC.person(A.EXTRA[f'fbl_{arm}']['all_loso'], 'all_loso').mean():.2f}", flush=True)
fams = [f for f in A.FAMS if f != "sbnfbl"]; assert A.FAM["sbnfbl"] == [(8, 1.0)], A.FAM["sbnfbl"]
R = []
for bn in env.BASES:
    s = {arm: SC.person(A.system(bn, fams, extra=[(f"fbl_{arm}", 1.0)])["out"]["all_loso"], "all_loso") for arm in ("twin", "ssl")}
    R.append(s["ssl"] - s["twin"]); print(f"{bn}: ssl against twin, new {R[-1].mean():+.2f}", flush=True)
d = np.mean(R, 0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90]); k = int(np.argmax(d))
print(f"MEAN ssl against twin, new {d.mean():+.2f} person 80% [{bt[0]:+.2f}, {bt[1]:+.2f}] people>=0 {(d >= 0).sum()}/10 | without best (P{k + 1:02d} {d[k]:+.1f}) {np.delete(d, k).mean():+.2f} | " + " ".join(f"{x:+.1f}" for x in d))
