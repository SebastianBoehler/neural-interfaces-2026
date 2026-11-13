"""Stream a package over the parity cases as the official evaluator does and record every window's outputs.

Usage: python lean/tools/record.py PKG_DIR OUT.npz --public proteus_windows.npz --external windows.npz [--cases a,b]

Every case runs on ``copy.deepcopy`` of the loaded model after ``reset_state()``, one window per ``predict`` call
(``benchmark_utils/streaming.py``). Cases:
  public/<person|session>   every public session, in file order (recording order);
  unknown/<person|session>  sessions of external people (nobody to recognise);
  switch                    half a session of one public person, then a session of another, without a reset;
  long                      three sessions of one person as one stream (more windows than any public session);
  stress                    a public session with corrupted windows (NaN, inf, zeros, flat channel, huge, constant);
  perm, missing             a public session delivered in another montage order / with two channels absent.
Recorded per window: label, final posterior, member mix, core1 posterior, 256-PC view, unrouted posterior, the two
gates' (person, score, ramp), the own-space stage's (person, ramp, log members) and the wall time of ``predict``.
"""
import argparse
import copy
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

FIELDS = {"label": 1, "out": 3, "mix": 3, "core1": 3, "shadow": 3, "u": 256, "contrast": 3, "identity": 3, "stage": 5}
SWITCH = ("Dreyer2026Proteus/P04|2", 95, "Dreyer2026Proteus/P10|2")
LONG = ("Dreyer2026Proteus/P10|1", "Dreyer2026Proteus/P10|2", "Dreyer2026Proteus/P10|3")
STRESS, MONTAGE, N_UNKNOWN = "Dreyer2026Proteus/P04|3", "Dreyer2026Proteus/P02|1", 4


def load_windows(path):
    with np.load(path) as f:
        d = {k: f[k] for k in ("X", "subject", "session", "ch_names")}
    d["stream"] = np.char.add(np.char.add(d["subject"], "|"), d["session"])
    return d


def load_model(pkg, ch_names, tag="pkg"):
    """The package's model as the harness builds it; the base solver is stubbed (no Benchopt needed here)."""
    if "benchmark_utils.base_solver" not in sys.modules:
        base = types.ModuleType("benchmark_utils.base_solver")
        base.CompetSolver = object
        sys.modules.update({"benchmark_utils": types.ModuleType("benchmark_utils"), "benchmark_utils.base_solver": base})
    spec = importlib.util.spec_from_file_location(tag, Path(pkg) / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Solver.__new__(mod.Solver).load_model({"submission_dir": Path(pkg), "device": "cpu", "ch_names": ch_names})


def tap(model):
    """Function returning the last window's internals: the lean model's ``trace``, or a tap on the original's
    routing step (submissions/ensemble_v5b)."""
    if hasattr(model, "trace"):
        return lambda: model.trace
    rec, route = {}, model.route
    step = route.step

    def tapped(x, u, mix, core1, qs):
        out = step(x, u, mix, core1, qs)
        rec.update(u=u, mix=mix, core1=core1, shadow=qs, out=out, contrast=route.last, identity=route.bio.last,
                   stage=None if route.stage is None else (*route.stage[:2], *route.stage[2]))
        return out
    route.step = tapped
    return lambda: rec


def run_stream(model, X):
    """One stream on a fresh copy, window by window; a row of FIELDS plus seconds per window."""
    cur = copy.deepcopy(model)
    cur.reset_state()
    read, rows = tap(cur), {k: [] for k in (*FIELDS, "seconds")}
    for i in range(len(X)):
        x = torch.from_numpy(X[i:i + 1])
        t0 = time.perf_counter()
        label = np.asarray(cur.predict(x))
        rows["seconds"].append(time.perf_counter() - t0)
        assert label.shape == (1,)
        rows["label"].append(label)
        for k, n in FIELDS.items():
            if k != "label":
                v = read()[k]
                rows[k].append(np.full(n, np.nan) if v is None else np.asarray(v, np.float64).reshape(n))
    return {k: np.asarray(v) for k, v in rows.items()}


def corrupt(X):
    """Fixed corruptions for the stress case (window index: what)."""
    X = X.copy()
    X[5, 3, 100:110] = np.nan
    X[12] = 0.0
    X[40, 7, 50] = np.inf
    X[41, 9, 60] = -np.inf
    X[70, 20] = X[70, 20, 0]
    X[100] *= 1e6
    X[130] = 1.0
    X[131] = np.nan
    return X


def cases(pub, ext):
    """Yield (name, windows, channel names) for every parity case."""
    names, rows = [str(c) for c in pub["ch_names"]], lambda d, s: d["X"][d["stream"] == s]
    for s in dict.fromkeys(pub["stream"]):
        yield f"public/{s.split('/')[-1]}", rows(pub, s), names
    for s in list(dict.fromkeys(ext["stream"]))[:N_UNKNOWN]:
        yield f"unknown/{s.split('/')[-1]}", rows(ext, s), names
    yield "switch", np.concatenate([rows(pub, SWITCH[0])[:SWITCH[1]], rows(pub, SWITCH[2])]), names
    yield "long", np.concatenate([rows(pub, s) for s in LONG]), names
    yield "stress", corrupt(rows(pub, STRESS)), names
    order = np.arange(len(names))[::-1]
    yield "perm", rows(pub, MONTAGE)[:, order], [names[i] for i in order]
    yield "missing", rows(pub, MONTAGE), ["X1", "X2", *names[2:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("out")
    ap.add_argument("--public", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    pub, ext = load_windows(a.public), load_windows(a.external)
    assert [str(c) for c in ext["ch_names"]] == [str(c) for c in pub["ch_names"]]
    want, models, out, case_of = set(filter(None, a.cases.split(","))), {}, {}, []
    for name, X, ch in cases(pub, ext):
        if want and name not in want and name.split("/")[0] not in want:
            continue
        if tuple(ch) not in models:
            models[tuple(ch)] = load_model(a.pkg, ch, f"pkg{len(models)}")
        t0 = time.perf_counter()
        res = run_stream(models[tuple(ch)], np.ascontiguousarray(X, np.float32))
        for k, v in res.items():
            out.setdefault(k, []).append(v)
        case_of += [name] * len(X)
        print(f"{name:22s} windows {len(X):4d}  routed {int((np.nan_to_num(res['stage'][:, 1]) > 0).sum()):4d}  "
              f"{time.perf_counter() - t0:6.1f} s", flush=True)
    np.savez_compressed(a.out, case=np.asarray(case_of), **{k: np.concatenate(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
