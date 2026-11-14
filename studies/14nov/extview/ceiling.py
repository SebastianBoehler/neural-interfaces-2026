"""s233 step 1: each external expert alone on PROTEUS new people (all_loso rows; the experts never saw PROTEUS) and its
error correlation with the member mix (A.mix B2_pool, the slots of B2_pool|Ps|all_loso) within class x run, with the
eye net (recording-order oc_nomix, 3 seeds) as the reference of an independent view.
Usage: python studies/14nov/extview/ceiling.py mivma mavr  -> stdout (tee to runs/views_ext/ceiling.txt)."""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; p0 = "all_loso"; Z = A.L.Z
IDX = A.pl.B[f"idx|{p0}"].astype(int); y = Z[f"y|{p0}"].astype(int); per = Z[f"person|{p0}"]; sess = Z[f"sess|{p0}"]
W = np.load(REPO + "/data/proteus_windows.npz"); assert (W["y"][IDX] == y).all(), "replay rows are not the windows idx says"
run = np.char.add(sess.astype(str), np.char.add("|", np.char.add(W["task"][IDX].astype(str), np.char.add("|", W["run"][IDX].astype(str)))))
ps = np.char.add(W["subject"][IDX].astype(str), np.char.add("|", W["session"][IDX].astype(str)))
assert all(len(np.unique(ps[sess == s])) == 1 for s in np.unique(sess)) and len(np.unique(ps)) == len(np.unique(sess)), "session ids differ"
MIX = A.mix("B2_pool", p0, A.FAMS)[0]; LM = np.log(np.clip(MIX, 1e-6, 1))
OCU = np.log(np.clip(np.mean([np.load(f)["proba"] for f in sorted(glob.glob(f"{REPO}/runs/ocunet/probs/oc_nomix-{p0}-s*.npz"))], 0), 1e-3, 1))
NAMES = {0: "MI", 1: "SUB", 2: "WORD"}


def pba(pred, yy, pp, cls):
    return 100 * np.mean([np.mean([np.mean(pred[(pp == q) & (yy == c)] == c) for c in cls]) for q in np.unique(pp)])


def auc(s, yy):  # P(score of class 1 > class 0)
    r = np.argsort(np.argsort(s)) + 1; n1 = (yy == 1).sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def dcorr(a, b, g):
    a, b = a.astype(float).copy(), b.astype(float).copy()
    for u in np.unique(g):
        m = g == u; a[m] -= a[m].mean(); b[m] -= b[m].mean()
    return np.corrcoef(a, b)[0, 1]


def pair_report(tag, logit, a, b):
    """logit > 0 votes for class b; pair (a, b) of PROTEUS classes."""
    m = (y == a) | (y == b); yy = (y[m] == b).astype(int); s = logit[m]; pp = per[m]; g = np.char.add(run[m], y[m].astype(str))
    ba = pba((s > 0).astype(int), yy, pp, (0, 1)); sa = np.mean([auc(s[sess[m] == u], yy[sess[m] == u]) for u in np.unique(sess[m])])
    ba_q = [pba((s[pp == q] > 0).astype(int), yy[pp == q], pp[pp == q], (0, 1)) for q in np.unique(pp)]
    tx = np.where(yy == 1, s, -s)  # expert margin toward the true class
    tm = np.where(yy == 1, LM[m][:, b] - LM[m][:, a], LM[m][:, a] - LM[m][:, b]); to = np.where(yy == 1, OCU[m][:, b] - OCU[m][:, a], OCU[m][:, a] - OCU[m][:, b])
    ex, em, eo = tx < 0, tm < 0, to < 0
    print(f"  {tag:5s} {NAMES[a]}-vs-{NAMES[b]}: BA {ba:.1f} (people {' '.join(f'{v:.0f}' for v in ba_q)}), within-session AUC {sa:.3f} | "
          f"pairwise acc: expert {100 - 100 * ex.mean():.1f} mix {100 - 100 * em.mean():.1f} eye {100 - 100 * eo.mean():.1f} | corr with mix (class x run): margin {dcorr(tx, tm, g):+.3f}, errors {dcorr(ex, em, g):+.3f} | "
          f"eye-net vs mix: margin {dcorr(to, tm, g):+.3f}, errors {dcorr(eo, em, g):+.3f} | expert vs eye margin {dcorr(tx, to, g):+.3f}", flush=True)
    return ba, sa


for nm in sys.argv[1:]:
    z = np.load(f"{REPO}/runs/views_ext/{nm}.npz"); keys = [k for k in z.files if k != "classes"]
    print(f"{nm} ({' vs '.join(z['classes'])}): rows {z['rec'].shape}, orders {keys}", flush=True)
    for k in ("rec", "within0", "full0", "runs0"):
        lg = z[k][:, 1] - z[k][:, 0]; print(f" order {k}: logit mean {lg.mean():+.2f} sd {lg.std():.2f}", flush=True)
        if nm == "mivma":
            pair_report(k, lg, 0, 1)
        else:
            pair_report(k, lg, 0, 1); pair_report(k, lg, 2, 1)
    acc = [[pair_report("o" + k, z[k][:, 1] - z[k][:, 0], *pr) for pr in ([(0, 1)] if nm == "mivma" else [(0, 1), (2, 1)])] for k in keys]
    print(f" {nm}: mean over {len(keys)} orders (BA, AUC) per pair: {np.round(np.mean(acc, 0), 3).tolist()}", flush=True)
