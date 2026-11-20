"""GPU timing of the stack members and of the load-time self-test, inside the worker SIF.

usage: selftest_timing.py DIR OUT.json   DIR holds cap250.zip, hardcb_safe.zip, hardcb_safe_st.zip, hard_safe.zip,
hard_safe_st.zip and the repo's data/proteus_windows.npz is read from $REPO (default cwd).
1. load_model wall time per ZIP (CUDA, warm after the first), two rounds: the difference st - plain is the self-test.
2. warm per-window ms of CBMember.prob and hard_probability alone, on every public window in session order
   (state reset at each session start, hard actor reload included, 100 untimed warm-up windows first).
3. end-to-end paired check: cap250 vs hardcb_safe_st through fresh_model().predict on three sessions, ABAB."""
import copy
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

warnings.filterwarnings("ignore")
REPO = Path(os.environ.get("REPO", Path.cwd()))
sys.path.insert(0, str(REPO / "2026-competition"))
from benchmark_utils.streaming import fresh_model  # noqa: E402

DEV = os.environ.get("DEV", "cuda")
MAXW = int(os.environ.get("MAXW", 0))  # smoke only: keep the first MAXW windows
sync = torch.cuda.synchronize if DEV == "cuda" else (lambda: None)


def load(z, tag):
    s = TemporaryDirectory()
    ZipFile(z).extractall(s.name)
    spec = importlib.util.spec_from_file_location(f"sub_{tag}", Path(s.name) / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sync(); t = time.perf_counter()
    model = mod.Solver().load_model({"submission_dir": Path(s.name), "device": DEV})
    sync()
    return model, time.perf_counter() - t, s


def stats(ms):
    a = np.asarray(ms)
    return {"n": len(a), "mean_ms": a.mean(), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95))}


def main(d, out):
    d = Path(d)
    res = {"gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else DEV, "torch": torch.__version__}
    try:
        import braindecode
        res["braindecode"] = braindecode.__version__
    except Exception as e:
        res["braindecode"] = repr(e)
    names = ["hardcb_safe", "hardcb_safe_st", "hardcb_safe", "hardcb_safe_st", "hard_safe", "hard_safe_st"]
    res["load_s"], keep = [], {}
    for i, n in enumerate(names):
        m, t, s = load(d / f"{n}.zip", f"{n}{i}")
        res["load_s"].append((n, round(t, 3)))
        print("load", n, round(t, 3), flush=True)
        keep[n] = (m, s)
    with np.load(REPO / "data/proteus_windows.npz") as f:
        X, subj, sess = f["X"].astype(np.float32), f["subject"], f["session"]
    if MAXW:
        X, subj, sess = X[:MAXW], subj[:MAXW], sess[:MAXW]
    assert X.shape[1:] == (41, 480) and (MAXW or len(X) >= 2000), X.shape
    keys = [f"{a}/{b}" for a, b in zip(subj, sess)]
    order = sorted(set(keys), key=keys.index)
    ens = keep["hardcb_safe_st"][0]
    for x in X[:min(100, len(X))]:  # warm-up, untimed
        ens.cb.prob(x); ens.hard_probability(x)
    cb_ms, hard_ms = [], []
    for k in order:
        ens.cb.reset(); ens.hard_actors = None
        for i in np.flatnonzero(np.asarray(keys) == k):
            t = time.perf_counter(); ens.cb.prob(X[i]); sync(); t1 = time.perf_counter()
            ens.hard_probability(X[i]); t2 = time.perf_counter()
            cb_ms.append((t1 - t) * 1e3); hard_ms.append((t2 - t1) * 1e3)
    res["n_windows"], res["n_sessions"] = len(cb_ms), len(order)
    res["cb"], res["hard"] = stats(cb_ms), stats(hard_ms)
    res["cb_plus_hard_mean_ms"] = res["cb"]["mean_ms"] + res["hard"]["mean_ms"]
    print(json.dumps(res, indent=1, default=float), flush=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float))  # components kept even if the e2e step fails
    # end-to-end paired
    sel = [k for k in order if k in order[:3]]
    idx = [np.flatnonzero(np.asarray(keys) == k) for k in sel]
    models = {"cap250": load(d / "cap250.zip", "cap")[0], "stack": keep["hardcb_safe_st"][0]}

    def run(m):
        sync(); t = time.perf_counter(); n = 0; labels = []
        for ix in idx:
            c = fresh_model(m)
            for i in ix:
                labels.append(int(c.predict(torch.from_numpy(X[i:i + 1].copy()))[0])); n += 1
        sync()
        return time.perf_counter() - t, n, labels
    e2e = {"cap250": [], "stack": []}
    for arm in ("cap250", "stack", "cap250", "stack"):
        t, n, lab = run(models[arm]); e2e[arm].append(t); print("e2e", arm, round(t, 2), n, flush=True)
    res["e2e"] = {"sessions": sel, "n": n, "wall_s": e2e,
                  "added_ms_per_window": (min(e2e["stack"]) - min(e2e["cap250"])) / n * 1e3}
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["e2e"], indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
