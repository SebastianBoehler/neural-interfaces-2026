"""Test-time augmentation by time shifts, measured on the same trained nets.

For every fold a net is trained once, then predicts each test session with
and without TTA (mean softmax over circular shifts SHIFTS samples); both are
saved as NAME-PROTOCOL-sSEED.npz under NAME = CONFIG and CONFIG_tta, so
mix_probs.py can compare them member for member.
Usage: python experiments/tta_eval.py CONFIG PROTOCOLS SEED [SHIFTS]
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import protocol  # noqa: E402
import run_cv  # noqa: E402

cfg, protos, seed = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
shifts = tuple(int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "-6,-3,0,3,6").split(","))
run_cv.SEED[0] = seed
out = ROOT / os.environ.get("NI26_PROBS", "runs/experiments/probs")
out.mkdir(parents=True, exist_ok=True)
d = protocol.load()
for proto in protos:
    res = {False: [], True: []}
    idx_all = []
    for name, tr, te in protocol.folds(d, proto):
        m = run_cv.CONFIGS[cfg](d)
        m.fit(np.flatnonzero(tr))
        st = d["stream"][te]
        for s in dict.fromkeys(st):
            rows = np.flatnonzero(te & (d["stream"] == s))
            for tta in (False, True):
                m.tta = shifts if tta else ()
                res[tta].append(m.predict_proba_stream(rows))
            idx_all.append(rows)
        print(cfg, proto, name, flush=True)
    idx = np.concatenate(idx_all)
    order = np.argsort(idx)  # global row order, as protocol.evaluate saves
    for tta in (False, True):
        tag = f"{cfg}_tta" if tta else cfg
        np.savez(out / f"{tag}-{proto}-s{seed}.npz",
                 proba=np.concatenate(res[tta])[order], index=idx[order])
    for tta in (False, True):
        P = np.concatenate(res[tta])
        mask = np.zeros(len(d["y"]), bool)
        mask[idx] = True
        sc = protocol.score(d["y"][mask], P.argmax(1)[np.argsort(idx)], d, mask)
        print(f"{cfg} {proto} s{seed} tta={tta} cell={sc['balanced_accuracy_cell_mean']:.4f}", flush=True)
