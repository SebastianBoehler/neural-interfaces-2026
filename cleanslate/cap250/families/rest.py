"""Fit the v6 rest member (rest4.py arch sh, cond r1: task windows plus allowed Graz pre-cue windows, k 2)."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _aux_v6 import ROOT, SMOKE, family_args, load_module, load_source, require_cuda, start, write_fit

import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
from export_external_teacher import device_free_trace  # noqa: E402


def source_data(args):
    """(d, d2, sel): source rows only; rest rows are the pre-cue windows of Graz source rows, same session stream."""
    d = load_source(args.people, args.first_session_only, args.data)
    n = len(d["y"])
    with np.load(ROOT / "data/proteus_windows_start-4.5.npz") as r:
        for key in ("y", "subject", "session", "run"):
            if not (r[key][d["index"]] == d[key]).all():
                raise ValueError(f"pre-cue window file rows differ from the source rows in {key}")
        pre = r["X"][d["index"]]
    sel = np.flatnonzero(np.char.endswith(d["task"].astype(str), "Graz"))
    if not len(sel):
        raise ValueError("no Graz source rows: the rest class would be empty")
    d2 = {}
    for key, v in d.items():
        d2[key] = np.concatenate([v, v[sel]]) if isinstance(v, np.ndarray) and len(v) == n and key != "ch_names" else v
    d2["X"] = np.concatenate([d["X"], pre[sel].astype(d["X"].dtype)])
    d2["y"] = np.concatenate([d["y"], np.full(len(sel), 3, d["y"].dtype)])
    return d, d2, sel


def fit(args):
    require_cuda()
    rest4 = load_module("rest4", "studies/11nov/rest4/rest4.py")
    d, d2, sel = source_data(args)
    n, seed, t0 = len(d["y"]), args.seed, start()
    m = rest4.make("sh", "r1", d2, seed)
    if SMOKE:
        m.epochs = 1
    m.fit(np.arange(len(d2["y"])))
    m.set_k(2.0)
    m.net.cpu().eval()
    m.dev = "cpu"
    p = dict(align=np.bool_(m.align), k=np.float64(m.k), G=np.asarray(m.G, np.float64), scale=np.float64(m.scale), shrink=np.float64(0.001))
    if not (m.align is True and not m.ref_clip and not m.ref_band):
        raise ValueError("the package InputStream has no clip / band reference")
    rows = np.flatnonzero(d["stream"] == d["stream"][0])
    xd = m._aligned(rows, True)
    cs, count, out = 0.0, 0, []
    for r in rows:  # the package InputStream: shrunk covariance, running mean blended with k pseudo-windows of G
        x = d["X"][r].astype(np.float64)
        xc = x - x.mean(-1, keepdims=True)
        c = xc @ xc.T / x.shape[-1]
        c = 0.999 * c + 0.001 * np.trace(c) / len(c) * np.eye(len(c))
        count += 1
        cs = cs + c
        w, v = np.linalg.eigh((p["k"] * p["G"] + cs) / (p["k"] + count))
        out.append(((v / np.sqrt(np.maximum(w, 1e-12))) @ v.T @ x).astype(np.float32))
    da = float(np.abs(np.stack(out) - xd).max())
    if da >= 1e-4 * max(1.0, float(np.abs(xd).max())):
        raise ValueError(f"input stream differs from dev alignment by {da}")
    trace = device_free_trace(SB.to_stream(m.net, 2.0), torch.from_numpy(np.ascontiguousarray(xd[:4])))
    SB.reset_stream(trace)
    ref = protocol.predict_streams(m, d, np.ones(n, bool), proba=True)
    worst = 0.0
    for s in dict.fromkeys(d["stream"]):
        r = np.flatnonzero(d["stream"] == s)
        SB.reset_stream(trace)
        with torch.no_grad():
            pr = torch.softmax(trace(torch.from_numpy(m._aligned(r, True)))[:, :3].double(), 1).numpy()
        worst = max(worst, float(np.abs(pr - ref[r]).max()))
    if worst >= 1e-4:
        raise ValueError(f"rest trace differs from the dev net by {worst}")
    SB.reset_stream(trace)
    trace.save(str(args.out / f"rest_s{seed}.pt"))
    np.savez(args.out / f"rest_s{seed}.npz", **p)
    write_fit(args.out, dict(seed=seed, rows=n, rest_rows=len(sel), input_diff=da, trace_vs_dev=worst,
                             in_sample_acc=float((ref.argmax(1) == d["y"]).mean()),
                             gpu=torch.cuda.get_device_name(), sec=round(time.time() - t0)))


if __name__ == "__main__":
    fit(family_args(__doc__, (0, 1, 2)))
