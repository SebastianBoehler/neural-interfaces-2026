"""v6 build decision (13 Nov): seconds per fine-tuning step through a shipped SessionBN trace on the GPU, with TorchScript's
default (profiling, optimising) executor and with torch.jit.optimized_execution(False); clean forward without grad,
GPU-drawn augmentation, StreamBN affine trainable, state zeroed per forward. Usage: python trace_exec_bench.py PKG"""
import json, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_cost_session import strong_dev  # noqa: E402

FILES = ["sbn_k2_s0", "sbn_fbcnet_raw_k20_s0", "sbn_fblight_ea_rot05_k5_s0", "sbn_sincshallow_ea_ptm7r_rot05_k20_s0"]


def bench(pkg, f, t, opt_exec):
    dev = "cuda"; m = torch.jit.load(str(Path(pkg) / f"{f}.pt"), map_location=dev)
    st = [b for n, b in m.named_buffers() if n.rsplit(".", 1)[-1] in ("s1", "s2", "n")]
    ps = [p for p in m.parameters() if p.requires_grad] + [b for n, b in m.named_buffers() if n.rsplit(".", 1)[-1] in ("w", "b")]
    for p in ps:
        p.requires_grad_(True)
    opt = torch.optim.Adam(ps, lr=5e-4); g = torch.Generator(device=dev).manual_seed(0)
    X = torch.randn(t, 41, 480, device=dev); lab = torch.randint(0, 3, (t,), device=dev); w = torch.rand(t, device=dev)

    def step():
        opt.zero_grad()
        for b in st:
            b.detach_(); b.zero_()
        with torch.no_grad():
            z = m(X)[:, :3]
        for b in st:
            b.detach_(); b.zero_()
        zs = m(strong_dev(X, g))[:, :3]
        ((w * F.cross_entropy(zs, lab, reduction="none")).sum() / w.sum() + F.kl_div(F.log_softmax(zs, 1), F.softmax(z, 1), reduction="batchmean")).backward(); opt.step()
    with torch.jit.optimized_execution(opt_exec):
        for _ in range(3):
            step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            step()
        torch.cuda.synchronize()
    return (time.time() - t0) / 10


for f in FILES:
    for t in (30, 110, 190):
        print(json.dumps(dict(net=f, t=t, default=round(bench(sys.argv[1], f, t, True), 4), no_opt=round(bench(sys.argv[1], f, t, False), 4))), flush=True)
