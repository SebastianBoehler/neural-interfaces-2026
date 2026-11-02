"""Riemannian external heads (overnight 2 Nov): a non-CNN view of the MI-transfer teacher.

Usage: python scripts/diag/riemann_heads.py OUT_NPZ [--max 40000] [--C 0.1]
Per m7 corpus (experiments/pretrain.load_corpus, EA per recording session): band-passed
covariances (8-13, 13-30, 4-40 Hz), tangent vectors at the identity, and a standardised
logistic regression over the corpus' own classes (at most MAX windows, stratified by
row order). The same features of the causally EA-aligned PROTEUS windows
(screen_external_heads.causal_ea) give 19 per-corpus head log-probs, saved as
``riem_ea-m7-s0`` ... ``-s5`` (deterministic, identical copies) for
eval_external_teacher.py, so ``riem`` combines with seed-averaged CNN families.
No PROTEUS label is used here.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts/diag")]
from features import bandpass, covariances, logm, upper  # noqa: E402
from pretrain import load_corpus  # noqa: E402
from screen_external_heads import causal_ea, d  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BANDS = ((8, 13), (13, 30), (4, 40))
M7 = "physionetmi,cho2017,lee2019_mi,dreyer2023,schirrmeister2017,shin2017a,shin2017b"


def tangent(X):
    out = []
    for lo, hi in BANDS:
        F = []
        for i in range(0, len(X), 2048):
            F.append(upper(logm(covariances(bandpass(X[i:i + 2048], lo, hi)))))
        out.append(np.concatenate(F))
    return np.concatenate(out, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--max", type=int, default=40000)
    ap.add_argument("--C", type=float, default=0.1)
    a = ap.parse_args()
    Fp = tangent(causal_ea(d["X"]))
    cols = []
    for c in M7.split(","):
        X, y, classes, n_subj = load_corpus(c, "ea")
        keep = np.linspace(0, len(y) - 1, min(a.max, len(y))).astype(int)
        F = tangent(X[keep])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000))
        m.fit(F, y[keep])
        acc = m.score(F, y[keep])
        cols.append(m.predict_log_proba(Fp))
        print(c, classes, n_subj, len(keep), f"train acc {acc:.3f}", flush=True)
        del X, F
    H = np.concatenate(cols, 1)
    np.savez_compressed(a.out, **{f"riem_ea-m7-s{s}": H for s in range(6)})
    print("heads", H.shape)


if __name__ == "__main__":
    main()
