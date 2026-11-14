"""s236 descriptors: label-free per-session descriptors from the first K windows delivered in each order.
  e1  agreement of the eye view's argmax (ocushuf posterior, seed mean) with v5.1b's chain (P0r, B2_pool) argmax
  e2  mean max posterior of the eye view
  a   alpha/mu peak: 8-13 Hz log power minus the mean of the 4-7 and 14-20 Hz log powers (dB), median over the 41 channels
  s   slow band: 1-4 Hz power as a share of 1-40 Hz, median over channels
P0r comes from the order's reference file (REF, default runs/chainout/<order>.npz: v6f as shipped, equal to s231_<order>_w1.npz to 0.0).
Writes runs/sesstype/desc.npz: '<order>|K<k>' -> (14, 4) in session order of np.unique(sess), plus 'sess', 'person'.
Usage: [ORDERS=within0,...] [KS=20,40,80] python studies/14nov/sesstype/desc.py"""
import glob, os, sys
import numpy as np
from scipy.signal import welch
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import paths  # noqa: E402
B = np.load(paths.DATA + "/bases.npz", allow_pickle=True)
ORDERS = os.environ.get("ORDERS", "within0,within1,full0,full1,runs0,runs1").split(",")
KS = [int(k) for k in os.environ.get("KS", "20,40,80").split(",")]
NAMES = ("e1", "e2", "a", "s")
sess, person = B["sess|all_loso"], B["person|all_loso"]; N = len(sess); SESS = np.unique(sess)


def delivery_pos(order):
    """row -> position inside its session in the delivery order (pinned order file)."""
    sc, ps = order.rstrip("0123456789"), order[len(order.rstrip("0123456789")):]
    perm = np.load(REPO + "/runs/shuffle_perms.npz")[f"{sc}|{ps}|all_loso"]; assert sorted(perm) == list(range(N))
    pos = np.empty(N, int)
    for s in SESS:
        r = perm[sess[perm] == s]; pos[r] = np.arange(len(r))
    return pos


def spectra():
    with np.load(paths.WINDOWS) as w:
        X = w["X"][:N].astype(np.float64)
    f, P = welch(X, fs=120, nperseg=240, axis=-1)  # (N, 41, 121), 0.5 Hz bins
    return f, P


def main():
    f, P = spectra(); band = lambda lo, hi: (f >= lo) & (f <= hi)  # noqa: E731
    out = {"sess": SESS, "person": np.array([person[sess == s][0] for s in SESS]), "names": np.array(NAMES)}
    for o in ORDERS:
        ref = os.environ.get("REF", f"{REPO}/runs/chainout/{o}.npz"); z = np.load(ref)
        p0r = z["B2_pool|P0r"]
        ocs = sorted(glob.glob(f"{REPO}/runs/ocushuf/{o}/oc_nomix-all_loso-s*.npz")); assert ocs, o
        eye = np.mean([np.load(q)["proba"] for q in ocs], 0); assert eye.shape == (N, 3)
        pos = delivery_pos(o)
        for K in KS:
            D = np.zeros((len(SESS), len(NAMES)))
            for i, s in enumerate(SESS):
                r = np.flatnonzero((sess == s) & (pos < K)); assert len(r) == K, (o, s, len(r))
                Pm = P[r].mean(0)  # (41, F): mean spectrum of the first K delivered windows
                lg = np.log10(Pm) * 10
                alpha = lg[:, band(8, 13)].mean(1) - 0.5 * (lg[:, band(4, 7)].mean(1) + lg[:, band(14, 20)].mean(1))
                slow = Pm[:, band(1, 4)].sum(1) / Pm[:, band(1, 40)].sum(1)
                D[i] = [np.mean(eye[r].argmax(1) == p0r[r].argmax(1)), eye[r].max(1).mean(), np.median(alpha), np.median(slow)]
            out[f"{o}|K{K}"] = D
        print(o, "K40 per session (e1 e2 a s):", flush=True)
        for i, s in enumerate(SESS):
            print(f"  {out['person'][i]} s{s}: " + " ".join(f"{v:.3f}" for v in out[f"{o}|K40"][i]) if 40 in KS else "", flush=True)
    os.makedirs(REPO + "/runs/sesstype", exist_ok=True); np.savez(REPO + "/runs/sesstype/desc.npz", **out); print("-> runs/sesstype/desc.npz")


if __name__ == "__main__":
    main()
