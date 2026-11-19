"""Fit the owning four-class Rest model on the authenticated P10 source."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import session_bn as SB
from export_external_teacher import device_free_trace
from source_labels import read_training_labels


BASE = Path("/mnt/lustre/work/gehler/gfh527")
REPO = BASE / "neural-interfaces-2026"
DATA = REPO / "data/proteus_windows.npz"
PRECUE = REPO / "data/proteus_windows_start-4.5.npz"
VIEW = BASE / "ni26_js321_20261117_returning_view_fit_p10_1/source_view_P10_s23.npz"
FIT_SHA = "db255fd45af6d17fa4ba1829781bac5a92a08afa1bec4e5bf85266b56ea84790"
HELD_SHA = "8aaa3c2c7d59a3851fb0c6cb0ece879771484a9a0151885c645df7b889e439b2"
SOURCE_PERSON = "Dreyer2026Proteus/P10"
K_REST, K_SBN = 5., 2.


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def row_hash(rows):
    return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()


def source_data():
    with np.load(DATA, allow_pickle=False) as task, np.load(PRECUE, allow_pickle=False) as rest:
        people, visits = task["subject"], task["session"]
        keep = (people != SOURCE_PERSON) | (visits == "1")
        fit, held = np.flatnonzero(keep), np.flatnonzero(~keep)
        g = np.flatnonzero(keep & np.char.endswith(task["task"].astype(str), "Graz"))
        if (len(fit), len(g), len(held), int(((people == SOURCE_PERSON) & keep).sum())) != (2270, 1152, 385, 192):
            raise ValueError("wrong authenticated source windows")
        if row_hash(fit) != FIT_SHA or row_hash(held) != HELD_SHA:
            raise ValueError("source row bytes differ from the §321 basis")
        with np.load(VIEW, allow_pickle=False) as basis:
            np.testing.assert_array_equal(basis["idx_tr"], fit)
        for key in ("subject", "session", "run", "task", "interface", "ch_names"):
            np.testing.assert_array_equal(task[key], rest[key])
        y = read_training_labels(DATA, fit, len(people))
        precue_y = read_training_labels(PRECUE, g, len(people))
        np.testing.assert_array_equal(precue_y, y[np.searchsorted(fit, g)])
        source = np.r_[fit, g]
        d = {key: task[key][source] for key in ("subject", "session", "run", "split", "task", "interface", "onset")}
        d.update(X=np.concatenate((task["X"][fit], rest["X"][g])), ch_names=task["ch_names"],
                 y=np.r_[y, np.full(len(g), 3)], tag="", w=np.ones(len(source)), nolow=np.zeros(len(source), dtype=bool))
    d["stream"] = np.char.add(np.char.add(d["subject"], "|"), d["session"])
    if (d["X"].shape != (3422, 41, 480) or d["X"].dtype != np.float32
            or np.any((d["subject"] == SOURCE_PERSON) & (d["session"] != "1"))):
        raise ValueError("held source contaminates the Rest recipe")
    return d, len(fit), row_hash(g)


def train(d, seed, smoke):
    recipe = REPO / "studies/11nov/rest4/rest4.py"
    spec = importlib.util.spec_from_file_location("rest4", recipe)
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    m = r.make("sh", "r1", d, seed)
    if (m.dev, m.arch, m.nout, m.w_rest, m.k, m.epochs, m.group, m.align) != (
            "cuda", "shallow", 4, 1., K_REST, 40, 32, True) or m.ref_clip or m.ref_band:
        raise ValueError("the owning Rest architecture or training recipe differs")
    if smoke:
        m.epochs = 1
    steps = 0

    def count(opt, args, kwargs):
        nonlocal steps
        if not isinstance(opt, torch.optim.AdamW):
            raise ValueError("unexpected optimizer in the source fit")
        steps += 1

    rows = np.arange(len(d["y"]))
    hook = register_optimizer_step_post_hook(count)
    try:
        m.fit(rows)
    finally:
        hook.remove()
    expected = m.epochs * len(m._batches(rows, np.random.default_rng(0)))
    if not steps or steps != expected or not m.n_sbn:
        raise ValueError("native Rest optimizer or SessionBN was inactive")
    np.testing.assert_array_equal(m.G, m.C.mean(0))
    if (m.G.shape != (41, 41) or not np.isfinite(m.G).all() or np.linalg.eigvalsh(m.G).min() <= 0
            or not np.isfinite(m.scale) or m.scale <= 0):
        raise ValueError("invalid authenticated source Rest prior")
    m.set_k(K_SBN)
    m.net.cpu().eval()
    m.dev = "cpu"
    return m, steps


def package_alignment(m, x):
    total = np.zeros_like(m.G)
    eye = np.eye(len(total))
    out = []
    for i, row in enumerate(x.astype(np.float64), 1):
        centered = row - row.mean(-1, keepdims=True)
        c = centered @ centered.T / row.shape[-1]
        total += .999 * c + .001 * np.trace(c) / len(c) * eye
        w, v = np.linalg.eigh((K_REST * m.G + total) / (K_REST + i))
        out.append(((v / np.sqrt(np.maximum(w, 1e-12))) @ v.T @ row).astype(np.float32))
    return np.stack(out)


@torch.inference_mode()
def run_trace(trace, x):
    SB.reset_stream(trace)
    logits = trace(torch.from_numpy(x))
    if logits.shape != (len(x), 4):
        raise ValueError("Rest trace has the wrong output shape")
    return torch.softmax(logits[:, :3].double(), 1).numpy()


def verify_trace(m, d, task_count):
    task_rows = np.arange(task_count)
    sources = d["stream"][task_rows]
    own = task_rows[sources == f"{SOURCE_PERSON}|1"]
    other = task_rows[sources == sources[0]][:17]
    x = m._aligned(own, True)
    np.testing.assert_array_equal(x[:17], m._aligned(own[:17], True))
    raw_native = package_alignment(m, d["X"][own])
    align = float(np.abs(raw_native - x).max()) / max(1., np.abs(x).max().item())
    trace = device_free_trace(SB.to_stream(m.net, K_SBN), torch.from_numpy(np.ascontiguousarray(x[:4])))
    full = run_trace(trace, x)
    first = run_trace(trace, x[:17])
    run_trace(trace, m._aligned(other, True))
    np.testing.assert_array_equal(first, run_trace(trace, x[:17]))
    prefix = float(np.abs(full[:17] - first).max())
    np.testing.assert_array_equal(full[:17].argmax(1), first.argmax(1))
    worst = 0.
    for session in dict.fromkeys(sources):
        ix = task_rows[sources == session]
        expected = m.predict_proba_stream(ix)
        converted = run_trace(trace, m._aligned(ix, True))
        worst = max(worst, float(np.abs(expected - converted).max()))
    if not all(np.isfinite(u) for u in (align, worst, prefix)) or max(align, worst, prefix) >= 1e-4:
        raise ValueError(f"Rest deployment fit did not parity-check {(align, worst, prefix)}")
    SB.reset_stream(trace)
    return trace, dict(input_relative_max_abs=align, eager_max_abs=worst, causal_prefix_max_abs=prefix,
                       exact_reset=True, source_visits=len(set(sources)))


def run(seed, smoke):
    if ("A100" not in torch.cuda.get_device_name() or torch.__version__ != "2.14.1+cu126"
            or os.environ.get("SLURM_CPUS_PER_TASK") != "8"):
        raise ValueError("wrong pinned Rest fit backend")
    torch.set_num_threads(8)
    start = monotonic()
    d, task_count, rest_rows_sha = source_data()
    print("Genuine Rest source fit", task_count, len(d["y"]) - task_count, mpath(), flush=True)
    m, steps = train(d, seed, smoke)
    trace, check = verify_trace(m, d, task_count)
    out = Path("smoke" if smoke else "fits") / f"rest_s{seed}"
    with out.with_suffix(".pt").open("xb") as handle:
        torch.jit.save(trace, handle)
    with out.with_suffix(".npz").open("xb") as handle:
        np.savez(handle, align=np.bool_(m.align), k=np.float64(m.k), G=m.G, scale=np.float64(m.scale),
                 shrink=np.float64(.001))
    for suffix in (".pt", ".npz"):
        out.with_suffix(suffix).chmod(0o444)
    proof = dict(member="rest", seed=seed, smoke=smoke, epochs=m.epochs, steps=steps,
                 source=sha("fit_source.sha256"), task_rows=task_count, rest_rows=len(d["y"]) - task_count,
                 fit_rows_sha256=FIT_SHA, held_rows_sha256=HELD_SHA, rest_rows_sha256=rest_rows_sha,
                 held_labels_opened=False, held_waveforms_used_in_fit=False, trace=sha(out.with_suffix(".pt")),
                 prior=sha(out.with_suffix(".npz")), checks=check, backend=mpath(), elapsed=monotonic() - start)
    with out.with_suffix(".json").open("x") as handle:
        json.dump(proof, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    out.with_suffix(".json").chmod(0o444)
    print("PASS genuine source Rest", sha(out.with_suffix(".json")), proof, flush=True)


def mpath():
    return dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), node=os.environ["SLURMD_NODENAME"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.smoke)
