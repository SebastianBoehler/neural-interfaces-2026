"""M2 stage 0: ceiling of per-person experts. EA + 5-band tangent + std LR, LOPO P01-P10 (non-causal EA, ceiling only).
python moonM2/stage0.py  -> prints pooled, expert, oracle numbers; caches features in moonM2/feat.npz (untracked)."""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score as bacc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]
from features import logm, invsqrtm, upper  # noqa: E402

BANDS = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
CACHE = Path(__file__).with_name("feat.npz")


def features():
    d = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    cell = np.array([f"{a}|{b}|{c}" for a, b, c in zip(d["subject"], d["session"], d["interface"])])
    if CACHE.exists():
        return np.load(CACHE)["F"], d["y"], person, cell
    X = d["X"].astype(np.float64)
    F = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], btype="band", fs=120, output="sos")
        Xb = sosfiltfilt(sos, X, axis=-1)
        C = np.einsum("nct,ndt->ncd", Xb, Xb) / Xb.shape[-1]
        C += 1e-3 * np.trace(C, axis1=1, axis2=2)[:, None, None] / 41 * np.eye(41)
        out = np.zeros((len(C), 41 * 42 // 2), np.float32)
        for c in np.unique(cell):
            m = cell == c
            Pm = invsqrtm(C[m].mean(0))  # EA reference (whole session: ceiling only)
            out[m] = upper(logm(Pm @ C[m] @ Pm))
        F.append(out)
    F = np.concatenate(F, 1)
    np.savez(CACHE, F=F)
    return F, d["y"], person, cell


def cellmean(y, p, cell, mask):
    return np.mean([bacc(y[mask & (cell == c)], p[mask & (cell == c)]) for c in np.unique(cell[mask])])


def fit(F, y, idx, C=0.01):
    sc = StandardScaler().fit(F[idx])
    lr = LogisticRegression(C=C, max_iter=300).fit(sc.transform(F[idx]), y[idx])
    return lambda Z: lr.predict_proba(sc.transform(Z))


def main():
    F, y, person, cell = features()
    P = sorted(set(person))
    rows = []
    for h in P:
        te = person == h
        pool = fit(F, y, np.flatnonzero(~te))(F[te])
        ex = {q: fit(F, y, np.flatnonzero(person == q))(F[te]) for q in P if q != h}
        yt, ct = y[te], cell[te]
        pm = np.ones(te.sum(), bool)
        cm = lambda p, m=pm: cellmean(yt, p.argmax(1), ct, m)  # noqa: E731
        eb = {q: cm(ex[q]) for q in ex}
        odd = np.arange(te.sum()) % 2 == 1
        sh = []
        for a in (odd, ~odd):  # choose on one half of windows, score on the other
            best = max(ex, key=lambda q: cm(ex[q], a))
            sh.append(cm(ex[best], ~a))
        sh3 = []
        for a in (odd, ~odd):  # top-3 mixture chosen on one half, scored on the other (selection-bias-free)
            tp = sorted(ex, key=lambda q: cm(ex[q], a), reverse=True)[:3]
            sh3.append(cm(sum(ex[q] for q in tp) / 3, ~a))
            sh3.append(cm(pool + sum(ex[q] for q in tp) / 3, ~a))
        top = sorted(eb, key=eb.get, reverse=True)
        t3 = sum(ex[q] for q in top[:3]) / 3
        rows.append([cm(pool), np.mean(list(eb.values())), max(eb.values()), np.mean(sh), cm(t3), cm(sum(ex.values())), cm(pool + t3), np.mean(sh3[0::2]), np.mean(sh3[1::2])])
        print(h, "pool %.1f meanexp %.1f oracle %.1f splithalf %.1f top3mix %.1f allmix %.1f pool+top3 %.1f sh-top3 %.1f sh-pool+top3 %.1f best %s" % (
            *[100 * v for v in rows[-1]], top[0]), flush=True)
    A = np.array(rows) * 100
    print("person-mean  pool meanexp oracle splithalf top3mix allmix pool+top3 sh-top3 sh-pool+top3\n", np.round(A.mean(0), 2))
    nc = np.array([len(set(cell[person == h])) for h in P])
    print("cell-weighted", np.round((A * nc[:, None]).sum(0) / nc.sum(), 2))


if __name__ == "__main__":
    main()
