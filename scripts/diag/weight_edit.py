"""Weight-space programming on the teacher's map (2 Nov; after Takehi, Teoh, Zhang and Khattab,
"Sparks of Weight-Space Programming", MIT CSAIL blog): an LLM reads the fitted weights and writes
an edit program, without gradients; the harness only scores it.

Usage: NI26_EDIT=path/edit.py python scripts/diag/weight_edit.py BASE_CACHE HEADS_NPZ OUT_JSON
       python scripts/diag/weight_edit.py BASE_CACHE HEADS_NPZ --show
The edit file defines ``edit(coef, b, names) -> (coef, b)`` on a fitted logistic map (coef 3 x d,
classes MI / SUB / WG; names "arch|corpus:class") and optionally ``feats(F, names) -> (F, names)``.
Every fold's map (LOSO for dla and al, dev rows outside P10's later sessions for ds) is fitted as in
eval_external_teacher.py, then edited; the edited teacher (EEGNeX + IFNet m7r, C 0.01) enters R19e at
w 5. Prints sel per seed half, the per-participant dla change against the unedited teacher and the
mean without P10. ``--show`` prints the map fitted on all dev participants (seeds 0-11) to read.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

for k, v in (("NI26_C", "0.01"), ("NI26_WTS", "5"), ("NI26_HALVES", "s05=0-5,s611=6-11,s011=0-11")):
    os.environ.setdefault(k, v)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_external_teacher as E  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

FAM = ["eegnex_ea-m7r", "ifnet_ea-m7r"]
spec = os.environ.get("NI26_EDIT")
if spec:
    s = importlib.util.spec_from_file_location("edit", spec)
    ED = importlib.util.module_from_spec(s)
    s.loader.exec_module(ED)
else:
    ED = None
state = {}


def features(H, fl, seeds):
    F = E.__dict__["_features0"](H, fl, seeds)[0]
    names = [f"{f.split('_')[0]}|{n}" for f in fl for n in H["names"]]
    if ED is not None and hasattr(ED, "feats"):
        F, names = ED.feats(F, names)
    state["names"] = names
    return [F]


def fit_predict(Fs, tr, te):
    lr = LogisticRegression(C=E.C, max_iter=3000, class_weight="balanced").fit(Fs[0][tr], E.y[tr])
    coef, b = lr.coef_.copy(), lr.intercept_.copy()
    if ED is not None:
        coef, b = ED.edit(coef, b, state["names"])
    z = Fs[0][te] @ coef.T + b
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


E._features0, E.features, E.fit_predict = E.features, features, fit_predict


def show(H):
    F = features(H, FAM, range(12))[0]
    lr = LogisticRegression(C=E.C, max_iter=3000, class_weight="balanced").fit(F[E.DEV], E.y[E.DEV])
    print(f"{'head':38s} {'MI':>7s} {'SUB':>7s} {'WG':>7s}")
    for n, c in zip(state["names"], lr.coef_.T):
        print(f"{n:38s} " + " ".join(f"{v:7.3f}" for v in c))
    print(f"{'intercept':38s} " + " ".join(f"{v:7.3f}" for v in lr.intercept_))


def main():
    if sys.argv[3] == "--show":
        return show(np.load(sys.argv[2]))
    sys.argv = sys.argv[:4] + ["nx+if=" + "+".join(FAM)]
    E.main()
    r = json.loads(Path(sys.argv[3]).read_text())["nx+if"]["s011"]["dev_loso_all"]["w5"]
    pp = r["per_participant_delta"]  # vs R19e; the unedited teacher's are in BASE_PP for comparison
    base = json.loads(os.environ.get("NI26_BASE_PP", "{}"))
    diff = {p: round(v - base.get(p, 0.0), 2) for p, v in sorted(pp.items())}
    no10 = [v for p, v in diff.items() if p != "P10"]
    print("s011 dla per participant vs", "unedited" if base else "R19e", diff,
          "| mean without P10", round(float(np.mean(no10)), 2), flush=True)


if __name__ == "__main__":
    main()
