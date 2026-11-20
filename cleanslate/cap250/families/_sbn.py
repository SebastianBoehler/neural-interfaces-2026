"""Shared SessionBN fit and stateful-trace export for the four cap250 sbn_* families.

Recipe of scripts/export_sbn.py: fit ``experiments/session_bn.make(cfg)`` on the source rows
(one fit per family and seed), then export one stateful TorchScript trace per latent prior k.
Package keys follow the sealed archive: align/k/G/scale from the fit, shrink 1e-3, sjit_file.
"""

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ASSETS, ROOT, family_args, load_source

sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import run_cv  # noqa: E402
from session_bn import make, reset_stream, STATE, to_stream  # noqa: E402

SHRINK = 1e-3
SMOKE_EPOCHS = 2


def device_free_trace(module, example):
    """Trace without baked device constants (copy of export_external_teacher.device_free_trace)."""
    to = torch.Tensor.to

    def same_device_noop(self, *args, **kw):
        if not kw and len(args) == 1 and isinstance(args[0], torch.device) and args[0] == self.device:
            return self
        return to(self, *args, **kw)

    torch.Tensor.to = same_device_noop
    try:
        with torch.no_grad():
            traced = torch.jit.trace(module, example, check_trace=False)
    finally:
        torch.Tensor.to = to
    assert 'value="cpu"' not in traced.inlined_graph.str(), "device constant left in the trace"
    return traced


def fit_net(cfg, data, seed):
    net = make(cfg, data, seed)
    if net.init:  # external m7r trunk: resolve under CAP250_ASSETS and fail loudly when absent
        net.init = net.init.replace(str(run_cv.ROOT), str(ASSETS))
        if not Path(net.init.format(seed=seed)).is_file():
            raise FileNotFoundError(net.init.format(seed=seed))
    if os.environ.get("CAP250_SMOKE") == "1":
        net.epochs = SMOKE_EPOCHS
    if not torch.cuda.is_available() and os.environ.get("CAP250_ALLOW_CPU") != "1":
        raise RuntimeError("SessionBN fits require CUDA (set CAP250_ALLOW_CPU=1 for a local smoke)")
    t0 = time.time()
    net.fit(np.arange(len(data["y"])))
    net.net.cpu().eval()
    net.dev = "cpu"
    return net, time.time() - t0


def export(net, data, k, path):
    """Trace the stream form of ``net`` at latent prior k, save it and check it against the eager net."""
    net.set_k(k)
    stream = data["stream"]
    sessions = list(dict.fromkeys(stream))
    first = np.flatnonzero(stream == sessions[0])
    H = net._aligned(first, causal=True)
    traced = device_free_trace(to_stream(net.net, k), torch.from_numpy(H[:4]))
    reset_stream(traced)
    traced.save(str(path))
    mod = torch.jit.load(str(path))
    buffers = [b for name, b in mod.named_buffers() if name.rsplit(".", 1)[-1] in STATE]
    if not buffers or len(buffers) != 3 * len(net.sbns()) or any(float(b.abs().sum()) for b in buffers):
        raise RuntimeError(f"{path.name}: stateful buffers missing or non-zero")
    worst = 0.0
    with torch.no_grad():
        for name in sessions[:2]:
            rows = np.flatnonzero(stream == name)
            reset_stream(mod)
            got = torch.softmax(mod(torch.from_numpy(net._aligned(rows, causal=True))).double(), 1).numpy()
            worst = max(worst, float(np.abs(got - net.predict_proba_stream(rows)).max()))
    if not worst < 1e-4:
        raise RuntimeError(f"{path.name}: trace differs from eager SessionBN by {worst}")
    return worst


def run(description, cfg, entries):
    """entries: ((package index base, trace prefix, latent prior k), ...); seeds 0-5."""
    args = family_args(description, range(6))
    data = load_source(args.people, args.first_session_only, args.data)
    net, seconds = fit_net(cfg, data, args.seed)
    arrays, diag = {}, {"cfg": cfg, "seed": args.seed, "fit_seconds": seconds,
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                        "n_rows": len(data["y"]), "trace_max_abs_diff": {}}
    for base, prefix, k in entries:
        i = base + args.seed
        name = f"{prefix}_s{args.seed}.pt"
        diag["trace_max_abs_diff"][name] = export(net, data, float(k), args.out / name)
        arrays.update({f"net{i}_align": np.array(bool(net.align)), f"net{i}_k": np.array(float(net.k)),
                       f"net{i}_G": np.asarray(net.G, np.float64), f"net{i}_scale": np.array(float(net.scale)),
                       f"net{i}_shrink": np.array(SHRINK), f"net{i}_sjit_file": np.array(name)})
    if not all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind == "f"):
        raise ValueError("nonfinite fitted SessionBN alignment array")
    np.savez(args.out / "weights.npz", **arrays)
    (args.out / "diagnostics").mkdir()
    (args.out / "diagnostics" / f"fit_s{args.seed}.json").write_text(json.dumps(diag, indent=1))
    print(json.dumps(diag), flush=True)
