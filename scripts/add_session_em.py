"""Add the per-session gaze-template stage (R19) to an exported package.

Usage: python scripts/add_session_em.py SRC_DIR OUT_DIR [--rows test|all]
Copies the package's trained weights with the current
``submissions/ensemble/submission.py`` and adds the ``em_*`` parameters
(defaults of experiments/session_em.SessionEM). Checks, streamed per session:
(1) the new code without the stage reproduces the source package exactly;
(2) with the stage it equals experiments.session_em.apply on the package's
own rebalanced member mix (the harness reference).
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import protocol  # noqa: E402
import session_em  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from build_sealed import em_params  # noqa: E402
from ensemble import rebalance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", choices=("test", "all"), default="all")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with np.load(a.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    assert not any(k.startswith("em_") for k in z), "source already has the stage"
    np.savez(a.out / "weights.npz", **z)
    shutil.copyfile(SRC, a.out / "submission.py")

    d = protocol.load()
    rows = d["split"] == "test" if a.rows == "test" else np.ones(len(d["y"]), bool)
    old = stream(load_solver(a.src, "old_pkg"), d, rows, lambda m, x: m.predict(x))
    off = load_solver(a.out, "new_off")
    P = stream(off, d, rows, lambda m, x: m.mixed_proba(x))
    same_off = float(np.mean(stream(off, d, rows, lambda m, x: m.predict(x)) == old))

    em = em_params({}, list(d["ch_names"]))
    np.savez(a.out / "weights.npz", **z, **em)
    on = stream(load_solver(a.out, "new_on"), d, rows, lambda m, x: m.predict(x))
    st = d["stream"][rows]
    Q = np.concatenate([rebalance(P[st == s], float(z["rb_alpha"]), float(z["rb_k"]), int(z["rb_w"]))
                        for s in dict.fromkeys(st)])
    F = session_em.gaze_features(d["X"][rows].astype(np.float64), list(d["ch_names"]))
    ref = session_em.apply(F, Q, st).argmax(1)
    res = {"source": str(a.src), "rows": a.rows, "n_windows": int(rows.sum()),
           "em": {k: v.tolist() for k, v in em.items()},
           "stage_off_matches_source": same_off,
           "stage_on_matches_harness": float(np.mean(on == ref)),
           "labels_changed_by_stage": int(np.sum(on != old)),
           "weights_sha256": hashlib.sha256((a.out / "weights.npz").read_bytes()).hexdigest(),
           "submission_sha256": hashlib.sha256((a.out / "submission.py").read_bytes()).hexdigest()}
    for name, pred in (("source_score", old), ("stage_score", on)):
        r = protocol.score(d["y"][rows], pred, d, rows)
        r.pop("cell_scores")
        res[name] = r
    (a.out.parent / f"{a.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert same_off == 1.0 and res["stage_on_matches_harness"] == 1.0


if __name__ == "__main__":
    main()
