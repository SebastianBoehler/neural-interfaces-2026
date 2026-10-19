"""Offline test of causal within-session self-training on top of the ensemble.

Usage: python scripts/selftrain_probs.py PROTOCOL [PROTOCOL ...]
For each fold, v2 is refitted on the fold's training windows; its causal
session features are standardised and projected onto 64 PCA components
fitted on the training windows. At window t (t >= T0) of a test session, a
shrinkage LDA is fitted on windows 0..t-1 with pseudo-labels (argmax of the
rebalanced ensemble probabilities, no true labels) and its probabilities are
mixed into the ensemble's with weight beta. Refitting every window is causal.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import protocol  # noqa: E402
import rebalance_probs as R  # noqa: E402
from models import TangentSpace  # noqa: E402

FB = ("theta", "alpha", "beta1", "beta2", "gamma", "high")
V2 = dict(bands=("delta",) + FB, means="center", slow="wave1")
MEMBERS = ("v2", "shallow_ea", "shallow_raw", "fbcnet_ea")
T0, BETAS = 30, (0.0, 0.2, 0.35, 0.5)


def session_feats(d, tr, te):
    m = TangentSpace(d, **V2)
    m.fit(np.flatnonzero(tr))
    scaler = m.clf[0]
    Ftr = scaler.transform(np.concatenate(
        [m._session_feats(np.flatnonzero(tr & (d["stream"] == s)), False)
         for s in dict.fromkeys(d["stream"][tr])]))
    pca = PCA(64, random_state=0).fit(Ftr)
    out = {}
    for s in dict.fromkeys(d["stream"][te]):
        idx = np.flatnonzero(te & (d["stream"] == s))
        out[s] = (idx, pca.transform(scaler.transform(m._session_feats(idx, True))))
    return out


def selftrain(Q, Z):
    """Per-window session-LDA probabilities from pseudo-labels of earlier windows."""
    S = np.full_like(Q, 1.0 / Q.shape[1])
    lab = Q.argmax(1)
    for t in range(T0, len(Q)):
        y = lab[:t]
        if len(set(y)) < Q.shape[1]:
            continue
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z[:t], y)
        S[t] = lda.predict_proba(Z[t:t + 1])[0]
    return S


def main():
    d = protocol.load()
    for proto in sys.argv[1:]:
        parts = [R.member(n, proto) for n in MEMBERS]
        P, idx = np.mean([p for p, _ in parts], 0), parts[0][1]
        row = {int(i): r for r, i in enumerate(idx)}
        Q = np.zeros_like(P)
        out = {b: np.zeros(len(idx), int) for b in BETAS}
        for _, tr, te in protocol.folds(d, proto):
            for s, (sidx, Z) in session_feats(d, tr, te).items():
                rows = np.array([row[int(i)] for i in sidx])
                q = R.rebalance(P[rows], d["stream"][sidx], 0.5, 5.0)
                S = selftrain(q, Z)
                for b in BETAS:
                    out[b][rows] = ((1 - b) * q + b * S).argmax(1)
        mask = np.zeros(len(d["y"]), bool)
        mask[idx] = True
        for b in BETAS:
            sc = protocol.score(d["y"][idx], out[b], d, mask)
            print(f"{proto:12s} beta={b:.2f} cell={sc['balanced_accuracy_cell_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
