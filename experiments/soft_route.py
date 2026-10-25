"""Soft routing to personal models by causal participant posterior (R8).

P_t = (1 - lam) * G_t + lam * sum_p q_t(p) * personal_p(t), q_t = causal
session posterior of identify.py's classifier, personal_p = v2 trained on
participant p's training windows only, G = r6nf mix; then rebalancing.
seen  : dev_session (P10 s2-3; training = dev windows except those).
unseen: dev_loso_all folds (one dev participant held out entirely; P04/P09
        stay training-only).
Usage: python experiments/soft_route.py [LAM ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import identify as ide  # noqa: E402  (runs its own checks at import: keep it light)
import protocol  # noqa: E402
import run_cv  # noqa: E402
from ensemble import rebalance  # noqa: E402
from mix_probs import member  # noqa: E402
from models import TangentSpace  # noqa: E402

lams = [float(x) for x in sys.argv[1:] if x[0].isdigit()] or [0.0, 0.1, 0.2, 0.3]
d, subj = ide.d, ide.subj
DIRS = "probs,probs_cloud,probs_r5,probs_local,probs_r7".split(",")
fam = ["v2", "shallow_ea", "shallow_raw", "fbcnet_ea", "fbcnet_raw"]
five = {str(i) for i in range(5)}


def global_mix(proto):
    src = "all_loso" if proto == "dev_loso_all" else proto
    F = [member(f, None if f == "v2" else five, src, DIRS)[:2] for f in fam]
    return {j: p for j, p in zip(F[0][1], sum(f[0] for f in F) / 5)}


def routed(train, test_rows, G, lam):
    """Causal soft-routed probabilities for one test session."""
    X = ide.feats(np.flatnonzero(train))
    clf = ide.model().fit(X[train], subj[train])
    lp = clf.predict_log_proba(X[test_rows])
    cum = np.cumsum(lp, 0) / np.arange(1, len(test_rows) + 1)[:, None]
    q = np.exp(cum - cum.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    pers = np.zeros((len(test_rows), 3))
    for k, p in enumerate(clf.classes_):
        m = TangentSpace(d, **run_cv.V2)
        m.fit(np.flatnonzero(train & (subj == p)))
        pers += q[:, [k]] * m.predict_proba_stream(test_rows)
    g = np.stack([G[j] for j in test_rows])
    return {lam_: rebalance((1 - lam_) * g + lam_ * pers, 1, 5, 30) for lam_ in lam}


def run(proto):
    G = global_mix(proto)
    dev = (d["split"] != "test")
    if proto == "dev_session":
        held = (subj == "P10") & (d["session"] != "1")
        folds = [(dev & ~held, held)]
    elif proto == "seen_session":  # evaluation only
        folds = [(~h, h) for h in ((subj == p) & (d["session"] != "1") for p in ("P04", "P10"))]
    elif proto == "all_loso":  # evaluation only
        folds = [(subj != p, subj == p) for p in sorted(set(subj))]
    else:
        folds = [(subj != p, subj == p) for p in sorted(set(subj[dev]))]
    preds = {lam: {} for lam in lams}
    for train, te in folds:
        for s in dict.fromkeys(d["stream"][te]):
            rows = np.flatnonzero(te & (d["stream"] == s))
            out = routed(train, rows, G, lams)
            for lam in lams:
                preds[lam].update(zip(rows, out[lam].argmax(1)))
        print(proto, "fold done", flush=True)
    for lam in lams:
        rows = np.array(sorted(preds[lam]))
        mask = np.zeros(len(subj), bool)
        mask[rows] = True
        sc = protocol.score(d["y"][mask], np.array([preds[lam][j] for j in rows]), d, mask)
        extra = ""
        for p in (("P04", "P09") if proto == "all_loso" else ("P04", "P10") if proto == "seen_session" else ()):
            sub = mask & (subj == p)
            extra += f" {p}={100 * protocol.score(d['y'][sub], np.array([preds[lam][j] for j in rows])[sub[mask]], d, sub)['balanced_accuracy_cell_mean']:.2f}"
        print(f"{proto} lam {lam:.2f} cell={100 * sc['balanced_accuracy_cell_mean']:.2f}{extra}", flush=True)


PROTOS = [p for p in sys.argv[1:] if not p[0].isdigit()] or ["dev_session", "dev_loso_all"]
lams = [x for x in lams] if lams else [0.0, 0.2]
for proto in PROTOS:
    run(proto)
