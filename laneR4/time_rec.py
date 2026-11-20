"""Lane R4: runtime of keep-adapting packages on RECOGNISED sessions (the public replay and lane T's noise timing both miss this cost:
recognition retires FT, and noise is never recognised). usage: time_rec.py ZIP DEVICE [SESSIONS]
Real public windows (P01 s1, P04 s2, P07 s1, P10 s3 by default: "subject:session" list), full-shuffled with a fixed seed, one window per
predict through the official fresh_model path; one warm-up session first. Prints one JSON line per session: seconds, recognition age,
refits, steps, ft_sec, cp_stopped. The final-stack zip is run in the same job as the paired reference (call twice, same node)."""
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [str(ROOT / "2026-competition")]
from benchmark_utils.streaming import fresh_model  # noqa: E402


def load_solver(package, device):
    spec = importlib.util.spec_from_file_location("runtime_solver", package / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Solver().load_model({"submission_dir": package, "device": device})


zip_path, device = Path(sys.argv[1]), sys.argv[2]
sessions = (sys.argv[3] if len(sys.argv) > 3 else "P01:1,P04:2,P07:1,P10:3").split(",")
with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as d:
    streams = []
    for s in sessions:
        who, ses = s.split(":")
        sel = (d["subject"] == f"Dreyer2026Proteus/{who}") & (d["session"] == ses)
        X = d["X"][sel]
        order = np.random.default_rng(7).permutation(len(X))[:190]
        streams.append((s, torch.from_numpy(X[order].astype(np.float32))))
with tempfile.TemporaryDirectory() as t:
    zipfile.ZipFile(zip_path).extractall(t)
    tpl = load_solver(Path(t), device)

    def session(x):
        m = fresh_model(tpl)
        start = time.perf_counter()
        for i in range(len(x)):
            m.predict(x[i:i + 1])
        sec = time.perf_counter() - start
        info = dict(sec=round(sec, 2), n=len(x), rec=m.log6["rec"] if m.log6["n"] else None, refits=m.ft.refits, steps=m.ft.steps,
                    ft_sec=round(m.ft.ft_sec, 2), cp_stopped=getattr(m.cp, "stopped", None), recB=bool(m.recB))
        del m
        return info

    session(streams[0][1])  # warm-up (CUDA kernels, first refit)
    out = {s: session(x) for s, x in streams}
print(json.dumps(dict(zip=zip_path.name, device=device, sessions=out, total_s=round(sum(v["sec"] for v in out.values()), 2))), flush=True)
