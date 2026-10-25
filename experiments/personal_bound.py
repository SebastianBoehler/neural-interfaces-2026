"""Oracle bound for personal models on later sessions (R8).

For each held-out later session of a participant, a personal v2 is trained on
that participant's earlier (training) sessions only and mixed with the r6nf
global mix of the protocol: P = (1 - lam) * global + lam * personal, then the
usual causal rebalancing. "Oracle" = the participant is known; a real
package would have to recognise it without labels.
Usage: python experiments/personal_bound.py PROTOCOL [LAM ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
from ensemble import rebalance  # noqa: E402
from mix_probs import member  # noqa: E402
from models import TangentSpace  # noqa: E402

proto = sys.argv[1]
lams = [float(x) for x in sys.argv[2:]] or [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
d = protocol.load()
DIRS = "probs,probs_cloud,probs_r5,probs_local,probs_r7".split(",")
fam = ["v2", "shallow_ea", "shallow_raw", "fbcnet_ea", "fbcnet_raw"]
five = {str(i) for i in range(5)}
F = [member(f, None if f == "v2" else five, proto, DIRS)[:2] for f in fam]
idx = F[0][1]
G = sum(f[0] for f in F) / 5
pos = {j: i for i, j in enumerate(idx)}

personal = np.zeros_like(G)
for name, tr, te in protocol.folds(d, proto):
    subj = d["subject"][te][0]
    own = np.flatnonzero(tr & (d["subject"] == subj))  # earlier sessions only
    m = TangentSpace(d, **run_cv.V2)
    m.fit(own)
    st = d["stream"][te]
    for s in dict.fromkeys(st):
        rows = np.flatnonzero(te & (d["stream"] == s))
        personal[[pos[j] for j in rows]] = m.predict_proba_stream(rows)
    print(name, subj.split("/")[-1], "personal trained on", len(own), "windows", flush=True)

st = d["stream"][idx]
mask = np.zeros(len(d["y"]), bool)
mask[idx] = True
for lam in lams:
    P = (1 - lam) * G + lam * personal
    for s in dict.fromkeys(st):
        P[st == s] = rebalance(P[st == s], 1, 5, 30)
    pred = P.argmax(1)[np.argsort(idx)]
    out = {"all": protocol.score(d["y"][mask], pred, d, mask)["balanced_accuracy_cell_mean"]}
    for p in sorted({s[-3:] for s in d["subject"][idx]}):
        sub = mask & (d["subject"] == f"Dreyer2026Proteus/{p}")
        out[p] = protocol.score(d["y"][sub], P.argmax(1)[np.argsort(idx)][sub[mask]], d,
                                sub)["balanced_accuracy_cell_mean"]
    print(f"lam {lam:.2f}", " ".join(f"{k}={100 * v:.1f}" for k, v in out.items()), flush=True)
