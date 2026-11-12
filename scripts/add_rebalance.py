"""Set causal class rebalancing parameters of an exported ensemble package.

Usage: python scripts/add_rebalance.py SRC_DIR OUT_DIR [ALPHA] [K] [W]
Copies the package's trained weights with the current
``submissions/ensemble/submission.py``. Checks on every official test
window, streamed per session: (1) the new code with the source package's
own rebalancing parameters reproduces the source package's predictions;
(2) with ``rb_alpha``/``rb_k``/``rb_w`` set to ALPHA/K/W it equals
``experiments.ensemble.rebalance`` applied to the member mix. W = 0 uses all
earlier windows of the session.
"""

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))

import protocol  # noqa: E402
from ensemble import rebalance  # noqa: E402

SRC = ROOT / "submissions/ensemble/submission.py"


def load_solver(folder, tag, device=None):
    """device None: the package's default (cpu); "cuda" as the worker's harness passes it."""
    spec = importlib.util.spec_from_file_location(tag, folder / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = {"submission_dir": folder} if device is None else {"submission_dir": folder, "device": device}
    return mod.Solver.__new__(mod.Solver).load_model(meta)


def stream(model, d, te, fn):
    out, streams = [], d["stream"][te]
    for i, j in enumerate(np.flatnonzero(te)):
        if i == 0 or streams[i] != streams[i - 1]:
            model.reset_state()
        out.append(np.asarray(fn(model, d["X"][j:j + 1]))[0])
    return np.array(out)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    k = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    w = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    out.mkdir(parents=True, exist_ok=True)
    with np.load(src / "weights.npz") as f:
        z = {key: f[key] for key in f.files}
    np.savez(out / "weights.npz", **z)  # source parameters, new code
    shutil.copyfile(SRC, out / "submission.py")

    d = protocol.load()
    te = d["split"] == "test"
    old = stream(load_solver(src, "old_pkg"), d, te, lambda m, x: m.predict(x))
    off = load_solver(out, "new_src_params")
    P = stream(off, d, te, lambda m, x: m.mixed_proba(x))
    same_off = float(np.mean(stream(off, d, te, lambda m, x: m.predict(x)) == old))

    z.update(rb_alpha=np.float64(alpha), rb_k=np.float64(k), rb_w=np.int64(w))
    np.savez(out / "weights.npz", **z)
    on = stream(load_solver(out, "new_on"), d, te, lambda m, x: m.predict(x))
    streams = d["stream"][te]
    ref = np.concatenate([rebalance(P[streams == s], alpha, k, w).argmax(1)
                          for s in dict.fromkeys(streams)])
    res = {"source": str(src), "rb_alpha": alpha, "rb_k": k, "rb_w": w,
           "source_params_match_source_package": same_off,
           "on_matches_harness_rebalance": float(np.mean(on == ref)),
           "source_score": protocol.score(d["y"][te], old, d, te),
           "rebalanced_score": protocol.score(d["y"][te], on, d, te),
           "weights_sha256": hashlib.sha256((out / "weights.npz").read_bytes()).hexdigest()}
    for r in (res["source_score"], res["rebalanced_score"]):
        r.pop("cell_scores")
    (out.parent / f"{out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert same_off == 1.0 and res["on_matches_harness_rebalance"] == 1.0


if __name__ == "__main__":
    main()
