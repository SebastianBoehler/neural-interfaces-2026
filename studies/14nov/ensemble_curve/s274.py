"""s274 scoring (ledger s274): four new braindecode families and the EEGNet one-factor sweep, scored exactly as s271.py
(same cells, same people x cells bootstrap seed, same 3-seed new / 6-seed old convention). Default EEGNet = `eegn` (s271 files).
Selection caveat: the sweep configs are chosen on the cells and people they are read on; best-of-7 is reported next to the default.
Usage: python studies/14nov/ensemble_curve/s274.py"""
import itertools, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curve as C  # noqa: E402

ARCH = ("incep", "tcnet", "ifnet", "tsc"); SW = ("k30", "k60", "k90", "f16", "d4", "p25", "p50")
SWF = tuple(f"eegn_{t}" for t in SW); NEW = ARCH + ("eegn",) + SWF; OLD = ("fb", "fbl", "fbraw", "sinc")
cells = list(itertools.product(C.ORD, C.DRAWS)); nc = len(cells); N = {}
for o, d in cells:
    for f in OLD + NEW:
        for s in range(6 if f in OLD else 3):
            fn = f"{C.D}/ttt18{o}{d}_{f}_base-all_loso-s{s}.npz"; assert os.path.exists(fn), fn
            z = np.load(fn); assert z["proba"].shape == (len(C.y), 3), fn
            N[o, d, f, s] = z["proba"].astype(np.float64)
ref = np.load(f"{C.D}/ttt18within0d1_fb_base-all_loso-s0.npz")["index"]
for f in NEW: assert (np.load(f"{C.D}/ttt18within0d1_{f}_base-all_loso-s0.npz")["index"] == ref).all(), f
for k in list(N)[::11]:  # guard: 1-net subset equals own file BA
    assert np.allclose(C.subset_ba(N, k[0], k[1], [(k[2], k[3])]), C.person_ba(N[k]))
ba = lambda nets: np.stack([C.subset_ba(N, o, d, nets) for o, d in cells])  # (cells, people)
br = np.random.default_rng(1); idx = [(br.integers(0, len(C.PEOPLE), len(C.PEOPLE)), br.integers(0, nc, nc)) for _ in range(1000)]
nets = lambda fams, S: [(f, s) for f in fams for s in range(S)]


def rep(name, v):
    b = [v[np.ix_(oo, pp)].mean() for pp, oo in idx]; lo, hi = np.percentile(b, [10, 90])
    return f"  {name:50s} {v.mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]  people up {int((v.mean(0) > 0).sum())}/10  cells up {int((v.mean(1) > 0).sum())}/{nc}"


S3 = ("fb", "fbl", "sinc"); S4 = S3 + ("fbraw",); L = [f"s274 scoring; cells {cells}; {len(C.PEOPLE)} people; BA %, person-equal, mean posterior; marginal = BA(base + family 3 seeds) - BA(base); intervals 80 % people x cells",
    "old families 6 seeds (s0-5) unless stated; new families 3 seeds (s0-2). Default EEGNet = eegn (s271 files); p25 = the default again (replicate of identical config)", ""]
L.append("alone (3 nets, s0-2):")
for f in NEW + OLD: L.append(f"  {f:9s} {ba(nets([f], 3)).mean():.2f}")
M = {}
for bname, fams in (("{fb,fbl,sinc}", S3), ("{fb,fbl,sinc,fbraw}", S4)):
    base = ba(nets(fams, 6)); L += ["", f"marginal over base {bname} at old seeds=6: BA {base.mean():.2f}"]
    for f in NEW:
        M[bname, f] = ba(nets(fams, 6) + nets([f], 3)) - base; L.append(rep(f"+{f}", M[bname, f]))
L += ["", "equal-n swap: (base {fb,fbl,sinc} 3 seeds + family 3 seeds) - (base 4 seeds each, 12 nets):"]
b4 = ba(nets(S3, 4)); L.append(f"  base 12 nets BA {b4.mean():.2f}")
for f in NEW: L.append(rep(f"{f}", ba(nets(S3, 3) + nets([f], 3)) - b4))
L += ["", "mean Pearson correlation of posteriors with fb (flattened, mean over cells, seed-matched s0-2):"]
cor = lambda f, g, sh: np.mean([np.corrcoef(N[o, d, f, s].ravel(), N[o, d, g, (s + sh) % 3].ravel())[0, 1] for o, d in cells for s in range(3)])
L.append(f"  fb with fb (other seed, reference) {cor('fb', 'fb', 1):.3f}")
for f in NEW + ("fbl", "fbraw", "sinc"): L.append(f"  {f:9s} with fb {cor(f, 'fb', 0):.3f}")
L += ["", "EEGNet sweep: marginal of each config minus marginal of the default eegn (same base, 6 old seeds):"]
for bname in ("{fb,fbl,sinc}", "{fb,fbl,sinc,fbraw}"):
    L.append(f" base {bname}")
    for f in SWF: L.append(rep(f"{f} - eegn", M[bname, f] - M[bname, "eegn"]))
    cand = ("eegn",) + tuple(f for f in SWF if f != "eegn_p25")  # 7 distinct configs: default + 6 one-factor changes (p25 is the default)
    best = max(cand, key=lambda f: M[bname, f].mean())
    L.append(f"  best of 7 distinct configs by marginal (selected on these cells and people): {best}")
    L.append(rep(f"best ({best}) marginal, selected", M[bname, best])); L.append(rep(f"best ({best}) - default eegn, selected", M[bname, best] - M[bname, "eegn"]))
    L.append(rep("p25 - eegn (replicate; the noise floor of one config)", M[bname, "eegn_p25"] - M[bname, "eegn"]))
txt = "\n".join(L); print(txt)
os.makedirs(C.REPO + "/evidence/ensemble_curve_15nov", exist_ok=True); open(C.REPO + "/evidence/ensemble_curve_15nov/s274.txt", "w").write(txt + "\n")
