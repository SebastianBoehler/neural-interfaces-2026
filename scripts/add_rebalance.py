"""Add causal class rebalancing to an exported ensemble package.

Usage: python scripts/add_rebalance.py SRC_DIR OUT_DIR [ALPHA] [K]
Copies the package's trained weights, adds ``rb_alpha``/``rb_k`` and the
current ``submissions/ensemble/submission.py``. Checks on every official
test window, streamed per session: (1) the new code with rebalancing off
reproduces the source package's own predictions; (2) with rebalancing on it
equals ``experiments.ensemble.rebalance`` applied to the member mix.
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


def load_solver(folder, tag):
    spec = importlib.util.spec_from_file_location(tag, folder / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Solver.__new__(mod.Solver).load_model({"submission_dir": folder})


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
    out.mkdir(parents=True, exist_ok=True)
    with np.load(src / "weights.npz") as f:
        z = {key: f[key] for key in f.files}
    z.update(rb_alpha=np.float64(0.0), rb_k=np.float64(k))
    np.savez(out / "weights.npz", **z)
    shutil.copyfile(SRC, out / "submission.py")

    d = protocol.load()
    te = d["split"] == "test"
    old = stream(load_solver(src, "old_pkg"), d, te, lambda m, x: m.predict(x))
    off = load_solver(out, "new_off")
    P = stream(off, d, te, lambda m, x: m.mixed_proba(x))
    same_off = float(np.mean(stream(off, d, te, lambda m, x: m.predict(x)) == old))

    z["rb_alpha"] = np.float64(alpha)
    np.savez(out / "weights.npz", **z)
    on = stream(load_solver(out, "new_on"), d, te, lambda m, x: m.predict(x))
    streams = d["stream"][te]
    ref = np.concatenate([rebalance(P[streams == s], alpha, k).argmax(1)
                          for s in dict.fromkeys(streams)])
    res = {"source": str(src), "rb_alpha": alpha, "rb_k": k,
           "off_matches_source_package": same_off,
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
