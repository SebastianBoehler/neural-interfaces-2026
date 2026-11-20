"""Lane R4 zero-change on UNRECOGNISED streams: each arm zip against the final-stack zip, CPU, 60 seeded white-noise windows (recognition
never fires, FT refits at 20, 30, 40, 50 run), official fresh_model path, one window per predict. Requires bitwise equal scores for every
arm and recB False in all; R0z additionally must equal on a recognised real session (P04 s2, 60 shuffled windows).
usage: check_zero.py ZIP_BASE ZIP_ARM..."""
import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [str(ROOT / "2026-competition")]
from benchmark_utils.streaming import fresh_model  # noqa: E402


def load(zp):
    d = tempfile.mkdtemp()
    zipfile.ZipFile(zp).extractall(d)
    spec = importlib.util.spec_from_file_location(f"s_{Path(zp).stem}", Path(d) / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Solver().load_model({"submission_dir": Path(d), "device": "cpu"})


def run(tpl, x):
    m = fresh_model(tpl)
    lab, post = [], []
    for i in range(len(x)):
        lab.append(m.predict(x[i:i + 1]).numpy())
        post.append(np.asarray(m.trace["out"], dtype=np.float64))  # the step posterior (before count filter and CSP)
    return np.concatenate([np.stack(lab).reshape(len(x), -1), np.stack(post)], 1), bool(m.recB), m.ft.refits


with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as d:
    sel = (d["subject"] == "Dreyer2026Proteus/P04") & (d["session"] == "2")
    real = torch.from_numpy(d["X"][sel][np.random.default_rng(7).permutation(sel.sum())[:60]].astype(np.float32))
noise = torch.from_numpy((20 * np.random.default_rng(0).standard_normal((60, 41, 480))).astype(np.float32))
torch.set_num_threads(2)
base = load(sys.argv[1])
b_noise, b_real = run(base, noise), run(base, real)
assert not b_noise[1] and b_noise[2] == 5, ("base noise stream", b_noise[1:])
print("base: noise recB", b_noise[1], "refits", b_noise[2], "| real recB", b_real[1], "refits", b_real[2], flush=True)
bad = 0
for zp in sys.argv[2:]:
    t = load(zp)
    n, r = run(t, noise), run(t, real)
    same_noise = np.array_equal(n[0], b_noise[0])
    same_real = np.array_equal(r[0], b_real[0])
    print(Path(zp).name, "noise bitwise", same_noise, "refits", n[2], "recB", n[1], "| real bitwise", same_real, "refits", r[2], flush=True)
    bad += (not same_noise) + (n[1] or n[2] != 5)
    if "R0z" in zp:
        bad += not same_real
assert b_real[1], "real session was not recognised: instrument invalid"
if bad:
    raise SystemExit(f"ZERO-CHANGE FAILED ({bad})")
print("ZERO-CHANGE OK")
