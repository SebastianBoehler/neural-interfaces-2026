"""SEQ stage 0: oracle-structure Graz position prior through the route3geb chain replay."""
import sys, time, collections
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; M = A.M; SC = A.SC
z = np.load(env.REPO + "/data/proteus_windows.npz", allow_pickle=True)
y = z["y"]; per = np.array([s[-3:] for s in z["subject"]]); ses = z["session"]; task = z["task"]; run = z["run"]
N = len(y); pos = np.zeros(N, int)
for k in set(zip(per, ses, run)):
    m = np.flatnonzero((per == k[0]) & (ses == k[1]) & (run == k[2])); pos[m] = np.arange(len(m))
graz = np.char.endswith(task.astype(str), "Graz"); pm = pos % 3
PRI = np.ones((N, 3))
for p in np.unique(per):
    tr = graz & (per != p); T = np.zeros((3, 3))
    for m_ in range(3):
        T[m_] = np.bincount(y[tr & (pm == m_)], minlength=3) + 1.0
    T /= T.sum(1, keepdims=True)
    te = graz & (per == p); PRI[te] = 3 * T[pm[te]]
print("prior table (last person held out):\n", np.round(T, 3), flush=True)
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}
for p in A.PROTOS:
    assert (SC.B[f"y|{p}"] == y[IDX[p]]).all()
GAM = {"v": 0.0}
_rebal = M.rebal
def rebal(mix, proto, C):
    if GAM["v"] > 0:
        mix = mix * PRI[IDX[proto]] ** GAM["v"]; mix = mix / mix.sum(1, keepdims=True)
    return _rebal(mix, proto, C)
M.rebal = rebal

def cells(P, proto):
    """mean cell BA for Graz and BrainHero cells, person-equal."""
    yy, cell, pp = SC.B[f"y|{proto}"], SC.B[f"cell|{proto}"], SC.B[f"person|{proto}"]; g = graz[IDX[proto]]; pred = P.argmax(1)
    out = {}
    for name, sel in (("G", g), ("B", ~g)):
        acc = collections.defaultdict(list)
        for c in np.unique(cell[sel]):
            m = cell == c; acc[pp[m][0]].append(SC.cell_ba(yy[m], pred[m]))
        out[name] = np.mean([np.mean(v) for v in acc.values()])
    return out

def score(out):
    dn, ds = SC.person(out["all_loso"], "all_loso"), SC.person(out["seen_session"], "seen_session")
    cn, cs = cells(out["all_loso"], "all_loso"), cells(out["seen_session"], "seen_session")
    return 0.5 * dn.mean() + 0.5 * ds.mean(), dn, ds, cn, cs

bases = sys.argv[1].split(",") if len(sys.argv) > 1 else list(env.EXPL)
for bn in bases:
    GAM["v"] = 0.0; r0 = A.system(bn, A.FAMS); J0, dn0, ds0, cn0, cs0 = score(r0["out"])
    print(f"\n{bn} base J {J0:.3f} new {dn0.mean():.2f} seen {ds0.mean():.2f} | new Graz {cn0['G']:.2f} BH {cn0['B']:.2f} | seen Graz {cs0['G']:.2f} BH {cs0['B']:.2f}", flush=True)
    for inj in ("final", "mix"):
        for g in (0.5, 1.0, 1.5, 2.0):
            t0 = time.time()
            if inj == "final":
                out = {}
                for p in A.PROTOS:
                    q = r0["out"][p] * PRI[IDX[p]] ** g; out[p] = q / q.sum(1, keepdims=True)
            else:
                GAM["v"] = g; out = A.system(bn, A.FAMS)["out"]; GAM["v"] = 0.0
            J, dn, ds, cn, cs = score(out)
            print(f"  {inj:5s} g{g}: dJ {J - J0:+.2f} new {dn.mean() - dn0.mean():+.2f} ({int((dn >= dn0).sum())}/10) seen {ds.mean() - ds0.mean():+.2f} "
                  f"| new Graz {cn['G'] - cn0['G']:+.2f} BH {cn['B'] - cn0['B']:+.2f} | seen Graz {cs['G'] - cs0['G']:+.2f} BH {cs['B'] - cs0['B']:+.2f} "
                  f"| per-person new {np.round(dn - dn0, 1).tolist()} {time.time() - t0:.0f}s", flush=True)
