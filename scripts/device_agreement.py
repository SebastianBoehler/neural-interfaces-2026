"""Do a package's predictions on CUDA match its CPU predictions?

Usage: python scripts/device_agreement.py PKG_DIR
Streams every PROTEUS window per session (reset per session, as the
evaluator does) through the package loaded with meta["device"] = "cpu" and
= "cuda"; prints label agreement, max |dP| and per-window time per device.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import protocol  # noqa: E402

pkg = Path(sys.argv[1])
d = protocol.load()
res = {}
for dev in ("cpu", "cuda"):
    spec = importlib.util.spec_from_file_location(f"pkg_{dev}", pkg / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.Solver.__new__(mod.Solver).load_model({"submission_dir": pkg, "device": dev})
    out, st, t0 = [], d["stream"], time.perf_counter()
    for j in range(len(d["y"])):
        if j == 0 or st[j] != st[j - 1]:
            m.reset_state()
        out.append(m.mixed_proba(torch.from_numpy(d["X"][j:j + 1]))[0])
    res[dev] = (np.array(out), (time.perf_counter() - t0) / len(d["y"]) * 1e3)
a, b = res["cpu"][0], res["cuda"][0]
print(json.dumps({"package": str(pkg), "windows": len(a),
                  "label_agreement": float(np.mean(a.argmax(1) == b.argmax(1))),
                  "max_abs_prob_diff": float(np.abs(a - b).max()),
                  "ms_per_window": {k: round(v[1], 2) for k, v in res.items()}}))
