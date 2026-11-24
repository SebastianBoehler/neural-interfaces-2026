"""Lane P2: package cost on RECOGNISED sessions (real P04 / P10 windows, recognised at window ~10; the gate then retires FT).
usage: time_rec.py ZIP DEVICE ROOT: one warm-up session, then 4 sessions (P04 visits 1-2, P10 visits 1-2) of up to 190 real windows, official fresh_model per
session, one window per predict. One JSON line: seconds per session and window, [v6] recognition flag and the [disp] counters."""
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

zip_path, device, root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path[:0] = [str(root / "2026-competition")]
from benchmark_utils.streaming import fresh_model  # noqa: E402

with np.load(root / "data/proteus_windows.npz", allow_pickle=False) as z:
    x = torch.from_numpy(z["X"]); channels = z["ch_names"].tolist(); session = z["session"]
    subject = np.array([s.rsplit("/", 1)[-1] for s in z["subject"]])
sess = [np.flatnonzero((subject == p) & (session == s))[:190] for p in ("P04", "P10") for s in np.unique(session[subject == p])[:2]]
with tempfile.TemporaryDirectory() as t:
    zipfile.ZipFile(zip_path).extractall(t)
    spec = importlib.util.spec_from_file_location("runtime_solver", Path(t) / "submission.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tpl = mod.Solver().load_model({"submission_dir": Path(t), "device": device, "ch_names": channels})

    def run(rows):
        m = fresh_model(tpl); t0 = time.perf_counter()
        for i in rows:
            m.predict(x[i:i + 1])
        return time.perf_counter() - t0, dict(recB=bool(getattr(m, "recB", False)), refits=m.ft.refits, disp=dict(getattr(m, "disp_log", {})))
    run(sess[0])
    out = [run(r) for r in sess]
print(json.dumps(dict(zip=zip_path.name, device=device, per_session_s=[round(o[0], 2) for o in out],
                      ms_per_window=round(1000 * float(np.mean([o[0] / len(r) for o, r in zip(out, sess)])), 2), info=[o[1] for o in out])), flush=True)
