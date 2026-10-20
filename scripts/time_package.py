"""Time a package's inference per window, without data preparation.

Usage: python scripts/time_package.py PKG_DIR [PKG_DIR ...]
Streams the official-test windows one at a time through each package's
``Solver`` (``reset_state`` per session, as the evaluator does) and prints
the mean wall time of ``predict`` per window plus the host CPU. With two or
more packages it also asserts that their predictions agree on every window.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

import protocol  # noqa: E402
from add_rebalance import load_solver, stream  # noqa: E402


def main():
    d = protocol.load()
    te = d["split"] == "test"
    res, preds = {"host": platform.node(), "cpu": platform.processor(),
                  "cpus": os.cpu_count(), "load": os.getloadavg()}, []
    for i, pkg in enumerate(map(Path, sys.argv[1:])):
        model = load_solver(pkg, f"pkg{i}")
        t0 = time.perf_counter()
        preds.append(stream(model, d, te, lambda m, x: m.predict(x)))
        res[str(pkg)] = round(1e3 * (time.perf_counter() - t0) / te.sum(), 1)
    res["windows"] = int(te.sum())
    if len(preds) > 1:
        res["agreement"] = float(np.mean([np.mean(p == preds[0]) for p in preds]))
    print(json.dumps(res))
    assert res.get("agreement", 1.0) == 1.0


if __name__ == "__main__":
    main()
