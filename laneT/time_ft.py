"""Lane T: FT-stage cost on UNRECOGNISED sessions (the public replay retires FT at once for known people).
usage: time_ft.py ZIP DEVICE [N_SESSIONS]: one warm-up session, then N sessions of 190 white-noise windows (never recognised, so
FT refits run from window 20 to the cap), official fresh_model per session, one window per predict. Prints one JSON line:
seconds per window and per session, plus the refit/step counters of the [v6] report line."""
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
import importlib.util  # noqa: E402


def load_solver(package, device):  # native/tools/runtime_reference.load_solver, inlined (not in the Galvani checkout)
    spec = importlib.util.spec_from_file_location("runtime_solver", package / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Solver().load_model({"submission_dir": package, "device": device})


zip_path, device = Path(sys.argv[1]), sys.argv[2]
n_sess = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rng = np.random.default_rng(0)
with tempfile.TemporaryDirectory() as t:
    zipfile.ZipFile(zip_path).extractall(t)
    tpl = load_solver(Path(t), device)

    def session(x):
        m = fresh_model(tpl)
        start = time.perf_counter()
        for i in range(len(x)):
            m.predict(x[i:i + 1])
        out = time.perf_counter() - start
        ft = m.ft
        info = dict(refits=ft.refits, steps=ft.steps, capped=ft.capped, ft_sec=round(ft.ft_sec, 2), recB=bool(getattr(m, "recB", False)))
        del m
        return out, info

    mk = lambda: torch.from_numpy((20 * rng.standard_normal((190, 41, 480))).astype(np.float32))
    session(mk())
    secs, infos = zip(*[session(mk()) for _ in range(n_sess)])
print(json.dumps(dict(zip=zip_path.name, device=device, per_session_s=[round(s, 2) for s in secs],
                      ms_per_window=round(1000 * float(np.mean(secs)) / 190, 2), info=infos[0])), flush=True)
