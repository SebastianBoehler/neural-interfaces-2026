"""s279 B1 strengths scoring (own copy of s279.py for fb_am25 / fb_am (p 0.5) / fb_am100; fb family only)

Original doc: s279 B1 scoring (ledger s279): Fourier amplitude mixing across people, fb_am / eegn_am (3 seeds, SessionBN, from scratch) vs the stored
fb / eegn base files (same seeds s0-2, same cells). Code reused from s271.py. Person-equal BA of the mean posterior; 80 % interval resamples people
and (order, draw) cells (1000 resamples). Alone: am(3 nets) - plain(3 nets). Marginal over {fb,fbl,sinc} (and +fbraw), old families 3 and 6 seeds:
BA(base + am family added) - BA(base), next to the plain family's own marginal, plus the swap form (plain family replaced by its am twin).
Usage: python studies/14nov/ensemble_curve/s279.py"""
import itertools, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curve as C  # noqa: E402

OLD = ("fb", "fbl", "fbraw", "sinc"); AM = ("fb_am25", "fb_am", "fb_am100"); cells = list(itertools.product(C.ORD, C.DRAWS)); nc = len(cells)
N = {}
for o, d in cells:
    for f in OLD + AM:
        for s in range(6 if f in OLD else 3):
            fn = f"{C.D}/ttt18{o}{d}_{f}_base-all_loso-s{s}.npz"; assert os.path.exists(fn), fn
            z = np.load(fn); assert z["proba"].shape == (len(C.y), 3), fn
            N[o, d, f, s] = z["proba"].astype(np.float64)
ref = np.load(f"{C.D}/ttt18within0d1_fb_base-all_loso-s0.npz")["index"]
for f in AM: assert (np.load(f"{C.D}/ttt18within0d1_{f}_base-all_loso-s0.npz")["index"] == ref).all(), f
for k in list(N)[::5]: assert np.allclose(C.subset_ba(N, k[0], k[1], [(k[2], k[3])]), C.person_ba(N[k]))  # 1-net subset = own file BA
ba = lambda nets: np.stack([C.subset_ba(N, o, d, nets) for o, d in cells])  # (cells, people)
br = np.random.default_rng(1); idx = [(br.integers(0, len(C.PEOPLE), len(C.PEOPLE)), br.integers(0, nc, nc)) for _ in range(1000)]
nets = lambda fams, S: [(f, s) for f in fams for s in range(S)]


def rep(name, v):
    b = [v[np.ix_(oo, pp)].mean() for pp, oo in idx]; lo, hi = np.percentile(b, [10, 90])
    return f"  {name:62s} {v.mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]  people up {int((v.mean(0) > 0).sum())}/10  cells up {int((v.mean(1) > 0).sum())}/{nc}"



L = ["s279 B1 strengths (fb_am25 p 0.25, fb_am p 0.5, fb_am100 p 1.0; 3 seeds); BA % person-equal; 80 % interval over people and cells", ""]
b3 = ba(nets(("fb", "fbl", "sinc"), 3)); rule = {}
for a in AM:
    v = ba(nets([a], 3)) - ba(nets(["fb"], 3)); rule[a] = v
    L += [f"{a}: alone {ba(nets([a], 3)).mean():.2f} vs fb {ba(nets(['fb'], 3)).mean():.2f}", rep(f"{a} - fb alone", v),
          rep(f"swap {{{a},fbl,sinc}} - {{fb,fbl,sinc}}", ba(nets((a, "fbl", "sinc"), 3)) - b3),
          rep(f"add {{fb,{a},fbl,sinc}} - {{fb,fbl,sinc}}", ba(nets(("fb", a, "fbl", "sinc"), 3)) - b3),
          f"  per-net alone BA: " + " ".join(f"s{s} {np.mean([C.person_ba(N[o, d, a, s]).mean() for o, d in cells]):.2f}" for s in range(3)),
          f"  corr with fb {np.mean([np.corrcoef(N[o, d, a, s].ravel(), N[o, d, 'fb', s].ravel())[0, 1] for o, d in cells for s in range(3)]):.3f}"]
L += ["", "RULE (alone >= +1.0, people interval low > 0, >= 6/10 up -> PASS; < -0.5 CLOSED; else UNDECIDED):"]
for a, v in rule.items():
    lo = np.percentile([v[np.ix_(oo, pp)].mean() for pp, oo in idx], 10); up = int((v.mean(0) > 0).sum())
    L.append(f"  {a}: alone {v.mean():+.2f}, interval low {lo:+.2f}, people up {up}/10 -> " + ("PASS" if v.mean() >= 1.0 and lo > 0 and up >= 6 else ("CLOSED" if v.mean() < -0.5 else "UNDECIDED")))
txt = "\n".join(L); print(txt)
open(C.REPO + "/evidence/b1_15nov/s279_strengths.txt", "w").write(txt + "\n")
