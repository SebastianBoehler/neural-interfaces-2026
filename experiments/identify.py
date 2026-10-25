"""Can a session's participant be recognised causally, without labels? (R8)

Window features: tangent vectors of 7 filter-bank covariances at the global
training reference (no session centring, which would remove identity),
standardised, PCA, multinomial logistic regression over participants.
A session's posterior after t windows: softmax of the mean window
log-probabilities so far (causal).
seen   : train on all windows except P04/P10 sessions 2-3 (their session 1
         stays in training); test those four sessions: is the argmax right,
         how confident?
unseen : leave each participant out entirely; test its sessions: the max
         posterior should stay low, so a threshold can reject newcomers.
Usage: python experiments/identify.py [PCA_DIMS] [C]
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]
import protocol  # noqa: E402
from features import expm_sym  # noqa: E402
from models import TangentSpace  # noqa: E402

DIMS, C = 100, 0.1  # main() reads them from argv
T = (5, 10, 20, 40, 1000)
d = protocol.load()
subj = np.array([s[-3:] for s in d["subject"]])
FB = ("delta", "theta", "alpha", "beta1", "beta2", "gamma", "high")
ts = TangentSpace(d, bands=FB, align="none")


def feats(train_rows):
    ts.log_glob = [L[train_rows].mean(0) for L in ts.logs]
    ts.glob = [expm_sym(g) for g in ts.log_glob]
    return ts._features(np.arange(len(subj)), ts.glob)


def curves(clf, rows_by_session, X, truth=None):
    """Per session: (argmax correct?, max posterior) after each t in T."""
    out = []
    for rows in rows_by_session:
        lp = clf.predict_log_proba(X[rows])
        cum = np.cumsum(lp, 0) / np.arange(1, len(rows) + 1)[:, None]
        q = np.exp(cum - cum.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
        res = []
        for t in T:
            qt = q[min(t, len(rows)) - 1]
            ok = None if truth is None else clf.classes_[qt.argmax()] == truth
            res.append((ok, qt.max()))
        out.append(res)
    return out


def model():
    return make_pipeline(StandardScaler(), PCA(DIMS, random_state=0),
                         LogisticRegression(C=C, max_iter=3000))


def main():
    global DIMS, C
    DIMS = int(sys.argv[1]) if len(sys.argv) > 1 else DIMS
    C = float(sys.argv[2]) if len(sys.argv) > 2 else C
    later = (np.isin(subj, ["P04", "P10"])) & (d["session"] != "1")
    X = feats(np.flatnonzero(~later))
    clf = model().fit(X[~later], subj[~later])
    print(f"== seen (PCA {DIMS}, C {C}): correct / max posterior after t windows", T)
    for p in ("P10", "P04"):
        for se in ("2", "3"):
            rows = np.flatnonzero((subj == p) & (d["session"] == se))
            r = curves(clf, [rows], X, p)[0]
            print(p, "s" + se, " ".join(f"{'Y' if ok else 'n'}{q:.2f}" for ok, q in r))

    print("== unseen (leave participant out): max posterior after t windows", T)
    for p in sorted(set(subj)):
        tr = subj != p
        X = feats(np.flatnonzero(tr))
        clf = model().fit(X[tr], subj[tr])
        sess = [np.flatnonzero((subj == p) & (d["session"] == se))
                for se in sorted(set(d["session"][subj == p]))]
        for se, r in zip(sorted(set(d["session"][subj == p])), curves(clf, sess, X)):
            print(p, "s" + se, " ".join(f"{q:.2f}" for _, q in r))


if __name__ == "__main__":
    main()
