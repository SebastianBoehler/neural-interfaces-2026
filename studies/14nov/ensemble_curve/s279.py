"""s279 B1 scoring (ledger s279): Fourier amplitude mixing across people, fb_am / eegn_am (3 seeds, SessionBN, from scratch) vs the stored
fb / eegn base files (same seeds s0-2, same cells). Code reused from s271.py. Person-equal BA of the mean posterior; 80 % interval resamples people
and (order, draw) cells (1000 resamples). Alone: am(3 nets) - plain(3 nets). Marginal over {fb,fbl,sinc} (and +fbraw), old families 3 and 6 seeds:
BA(base + am family added) - BA(base), next to the plain family's own marginal, plus the swap form (plain family replaced by its am twin).
Usage: python studies/14nov/ensemble_curve/s279.py"""
import itertools, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curve as C  # noqa: E402

OLD = ("fb", "fbl", "fbraw", "sinc"); PLAIN = ("fb", "eegn"); AM = ("fb_am", "eegn_am"); cells = list(itertools.product(C.ORD, C.DRAWS)); nc = len(cells)
N = {}
for o, d in cells:
    for f in OLD + ("eegn",) + AM:
        for s in range(6 if f in OLD else 3):
            fn = f"{C.D}/ttt18{o}{d}_{f}_base-all_loso-s{s}.npz"; assert os.path.exists(fn), fn
            z = np.load(fn); assert z["proba"].shape == (len(C.y), 3), fn
            N[o, d, f, s] = z["proba"].astype(np.float64)
ref = np.load(f"{C.D}/ttt18within0d1_fb_base-all_loso-s0.npz")["index"]
for f in AM + ("eegn",): assert (np.load(f"{C.D}/ttt18within0d1_{f}_base-all_loso-s0.npz")["index"] == ref).all(), f
for k in list(N)[::5]: assert np.allclose(C.subset_ba(N, k[0], k[1], [(k[2], k[3])]), C.person_ba(N[k]))  # 1-net subset = own file BA
ba = lambda nets: np.stack([C.subset_ba(N, o, d, nets) for o, d in cells])  # (cells, people)
br = np.random.default_rng(1); idx = [(br.integers(0, len(C.PEOPLE), len(C.PEOPLE)), br.integers(0, nc, nc)) for _ in range(1000)]
nets = lambda fams, S: [(f, s) for f in fams for s in range(S)]


def rep(name, v):
    b = [v[np.ix_(oo, pp)].mean() for pp, oo in idx]; lo, hi = np.percentile(b, [10, 90])
    return f"  {name:62s} {v.mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]  people up {int((v.mean(0) > 0).sum())}/10  cells up {int((v.mean(1) > 0).sum())}/{nc}"


L = [f"s279 B1 scoring; cells {cells}; {len(C.PEOPLE)} people; BA % person-equal, mean posterior; 80 % interval over people and cells",
     "arms: fb_am, eegn_am (3 seeds s0-2); baselines: the stored fb, eegn files (s0-2); old families 3 seeds = s0-5 subsets", ""]
L.append("ALONE (3 nets each):")
for p, a in zip(PLAIN, AM):
    L.append(f"  {p} {ba(nets([p], 3)).mean():.2f}   {a} {ba(nets([a], 3)).mean():.2f}"); L.append(rep(f"{a} - {p} (alone)", ba(nets([a], 3)) - ba(nets([p], 3))))
L += ["", "per-net mean (seed-wise) alone BA: " + "  ".join(f"{f} s{s} {np.mean([C.person_ba(N[o, d, f, s]).mean() for o, d in cells]):.2f}" for f in PLAIN + AM for s in range(3))]
L += ["", "MARGINAL over the shipped base (new family 3 seeds ADDED; am arm and plain arm each against the same base):"]
for bname, fams in (("{fb,fbl,sinc}", ("fb", "fbl", "sinc")), ("{fb,fbl,sinc,fbraw}", ("fb", "fbl", "sinc", "fbraw"))):
    for S in (3, 6):
        base = ba(nets(fams, S)); L.append(f"base {bname} old seeds={S}: BA {base.mean():.2f}")
        for p, a in zip(PLAIN, AM):
            if p in fams: continue  # fb is in the base: the add form is the swap form below
            vp, va = ba(nets(fams, S) + nets([p], 3)) - base, ba(nets(fams, S) + nets([a], 3)) - base
            L += [rep(f"+{p}", vp), rep(f"+{a}", va), rep(f"+{a} minus +{p}", va - vp)]
L += ["", "SWAP form over {fb,fbl,sinc} at 3 seeds each (9 nets): plain family replaced by its am twin; and eegn_am vs eegn as a 10th..12th family:"]
b3 = ba(nets(("fb", "fbl", "sinc"), 3)); L.append(rep("{fb_am,fbl,sinc} - {fb,fbl,sinc}", ba(nets(("fb_am", "fbl", "sinc"), 3)) - b3))
L.append(rep("{fb,fb_am,fbl,sinc} (am added to fb) - {fb,fbl,sinc}", ba(nets(("fb", "fb_am", "fbl", "sinc"), 3)) - b3))
L += ["", "mean Pearson correlation of posteriors with the plain twin (seed-matched, mean over cells) and with fb:"]
cor = lambda f, g: np.mean([np.corrcoef(N[o, d, f, s].ravel(), N[o, d, g, s].ravel())[0, 1] for o, d in cells for s in range(3)])
L += [f"  fb_am with fb {cor('fb_am', 'fb'):.3f}   eegn_am with eegn {cor('eegn_am', 'eegn'):.3f}   eegn_am with fb {cor('eegn_am', 'fb'):.3f}   eegn with fb {cor('eegn', 'fb'):.3f}"]
a, e = (ba(nets([am], 3)) - ba(nets([pl], 3)) for pl, am in zip(PLAIN, AM))
L += ["", "RULE reading (alone >= +1.0, people interval above 0, >= 6/10 up -> marginal + chain screen; -0.5..+1.0 undecided; < -0.5 closed):"]
for nm, v in (("fb_am", a), ("eegn_am", e)):
    pb = [v[np.ix_(oo, pp)].mean() for pp, oo in idx]; lo = np.percentile(pb, 10); up = int((v.mean(0) > 0).sum())
    br_ = "PASS" if v.mean() >= 1.0 and lo > 0 and up >= 6 else ("CLOSED" if v.mean() < -0.5 else "UNDECIDED")
    L.append(f"  {nm}: alone {v.mean():+.2f}, interval low {lo:+.2f}, people up {up}/10 -> {br_}")
txt = "\n".join(L); print(txt)
os.makedirs(C.REPO + "/evidence/b1_15nov", exist_ok=True); open(C.REPO + "/evidence/b1_15nov/s279_b1.txt", "w").write(txt + "\n")
