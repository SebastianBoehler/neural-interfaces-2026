"""The dev SessionNet of a shipped SessionBN member, rebuilt from the package itself (review 13 Nov, item 5).

base_from_package(pkg, member_index, d) -> (m, info): experiments/session_bn.make(cfg, d, seed) with the trace's weights
(StreamBN rm/rv/w/b -> SessionBN running_mean/running_var/weight/bias), the member's alignment (net<i>_G, net<i>_scale,
net<i>_k from weights.npz) and the trace's own test prior k. Fails loudly unless (a) the package's causal input stream
equals m._aligned(causal=True) on a real session and (b) the eager net equals the shipped trace on those inputs.
So personal nets fine-tuned from m start from exactly the shipped member, not from a retrained twin
(scripts/export_sbn.trained() trains a new net when a checkpoint is missing)."""
import copy, hashlib, importlib.util, re, sys, types
from pathlib import Path
import numpy as np
import torch
import session_bn as SB
from deep import build

# trace file stem (without _s<seed>) -> (dev cfg, test prior k)
CFG = {"sbn_k2": ("fb_sbn_rot05", 2.0), "sbn_k5": ("fb_sbn_rot05", 5.0), "sbn_fbcnet_raw_k20": ("fbcnet_raw_sbn", 20.0),
       "sbn_fblight_ea_rot05_k5": ("fblight_ea_rot05_sbn", 5.0), "sbn_sincshallow_ea_ptm7r_rot05_k20": ("sincshallow_ea_ptm7r_rot05_sbn", 20.0)}
REN = {"running_mean": "rm", "running_var": "rv", "weight": "w", "bias": "b"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def base_from_package(pkg, i, d, tol=1e-5):
    pkg = Path(pkg); z = np.load(pkg / "weights.npz"); f = str(z[f"net{i}_sjit_file"])
    stem, seed = re.fullmatch(r"(.+)_s(\d+)\.pt", f).groups(); cfg, k = CFG[stem]; seed = int(seed)
    trace = torch.jit.load(str(pkg / f), map_location="cpu").eval()
    m = SB.make(cfg, d, seed); m.net = build(m.arch, d["X"].shape[1], d["X"].shape[2], m.pretrained, **m.net_kw); m._prepare()
    bn = {n for n, mod in m.net.named_modules() if isinstance(mod, SB.SessionBN)}; ts = dict(trace.state_dict()); new = {}
    for n in m.net.state_dict():
        pre, last = n.rsplit(".", 1)
        new[n] = ts[f"{pre}.{REN[last]}"] if pre in bn and last in REN else ts[n]
    m.net.load_state_dict(new); m.net.cpu().eval(); m.dev = "cpu"; m.set_k(k)
    m.G, m.scale = z[f"net{i}_G"].astype(np.float64) if f"net{i}_G" in z.files else m.G, float(z[f"net{i}_scale"])
    assert float(z[f"net{i}_k"]) == float(m.k), f"{f}: alignment prior k {float(z[f'net{i}_k'])} != dev {m.k}"
    # (a) the package's input stream equals the dev alignment on a real session; (b) eager equals the trace
    if "benchmark_utils.base_solver" not in sys.modules:  # the package's InputStream only; benchopt may be absent (Leonardo .venv)
        stub = types.ModuleType("benchmark_utils.base_solver"); stub.CompetSolver = object
        sys.modules.setdefault("benchmark_utils", types.ModuleType("benchmark_utils")); sys.modules["benchmark_utils.base_solver"] = stub
    spec = importlib.util.spec_from_file_location("pkgsub", pkg / "submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    p = {key[len(f"net{i}_"):]: z[key] for key in z.files if key.startswith(f"net{i}_")}
    rows = np.flatnonzero(d["stream"] == d["stream"][0]); st = mod.InputStream(p)
    Xp = np.stack([st(np.nan_to_num(np.asarray(d["X"][r], np.float32))) for r in rows]); Xd = m._aligned(rows, True)
    da = float(np.abs(Xp - Xd).max()); assert da < tol * max(1.0, float(np.abs(Xp).max())), f"{f}: package input stream differs from the dev alignment by {da}"
    for n_, b in trace.named_buffers():
        if n_.rsplit(".", 1)[-1] in SB.STATE:
            b.zero_()
    with torch.no_grad():
        X = torch.from_numpy(np.ascontiguousarray(Xp)); pt = torch.softmax(trace(X)[:, :3].double(), 1); pe = torch.softmax(m.net(X)[:, :3].double(), 1)
    dp = float((pt - pe).abs().max()); assert dp < tol, f"{f}: eager rebuild differs from the shipped trace by {dp}"
    return m, dict(trace=f, trace_sha256=sha(pkg / f), cfg=cfg, seed=seed, k=k, input_diff=da, eager_vs_trace=dp)


STEM = {"fb_sbn_rot05": "sbn_k2", "fbcnet_raw_sbn": "sbn_fbcnet_raw_k20", "fblight_ea_rot05_sbn": "sbn_fblight_ea_rot05_k5",
        "sincshallow_ea_ptm7r_rot05_sbn": "sbn_sincshallow_ea_ptm7r_rot05_k20"}  # dev cfg -> the shipped trace it was built from


def base(pkg, cfg, seed, d):
    """base_from_package for the member traced from dev cfg, seed (fb: its k2 trace); info["ckpt_sha256"] = the trace's"""
    z = np.load(Path(pkg) / "weights.npz"); want = f"{STEM[cfg]}_s{seed}.pt"
    i = next(i for i in range(int(z["n_nets"])) if f"net{i}_sjit_file" in z.files and str(z[f"net{i}_sjit_file"]) == want)
    m, info = base_from_package(pkg, i, d); info["ckpt_sha256"] = info["trace_sha256"]; return m, info


def graft_check(personal, shipped, n=48, tol=1e-5):
    """Max |posterior difference| between the shipped trace and the personal trace carrying the shipped output layer and
    StreamBN affine (everything pfh60 trains); > tol means the personal trace sits on another base."""
    a, b = copy.deepcopy(personal), copy.deepcopy(shipped)
    sp, sb = dict(b.named_parameters()), dict(b.named_buffers())
    with torch.no_grad():
        for name, t in a.named_parameters():
            if name.startswith("final_layer"):
                t.copy_(sp[name])
        for name, t in a.named_buffers():
            if name.rsplit(".", 1)[-1] in ("w", "b"):
                t.copy_(sb[name])
        g = torch.Generator().manual_seed(0); X = torch.randn(n, 41, 480, generator=g)
        out = []
        for mod in (a, b):
            for name, t in mod.named_buffers():
                if name.rsplit(".", 1)[-1] in SB.STATE:
                    t.zero_()
            out.append(torch.softmax(mod(X)[:, :3].double(), 1))
    return float((out[0] - out[1]).abs().max())
