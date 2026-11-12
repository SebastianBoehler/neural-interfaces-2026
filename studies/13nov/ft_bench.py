"""v5.3 runtime: cost of the fix recipe (ledger s101: weighted CE on augmented inputs + KL consistency, Adam) run
directly on the package's TorchScript SessionBN traces (eval-form StreamBN, state zeroed before every forward).
Per family (one seed) and session length t: seconds per step, eager loop and (CUDA) a captured CUDA graph of the step
at a padded size (rows past t are masked out; StreamBN is causal, so padding after the real windows changes nothing).
Usage: python studies/13nov/ft_bench.py PKG DEVICE   -> one JSON line per measurement"""
import json, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F

FAM = ["sbn_k2_s0", "sbn_fbcnet_raw_k20_s0", "sbn_fblight_ea_rot05_k5_s0", "sbn_sincshallow_ea_ptm7r_rot05_k20_s0"]
STATE = ("s1", "s2", "n")


def strong(X, g, shift=15, noise=0.2, drop=0.1):
    sh = int(torch.randint(-shift, shift + 1, (1,), generator=g)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g).to(X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g) > drop).float().to(X.device); return Xs * keep


def setup(pkg, f, dev):
    m = torch.jit.load(str(Path(pkg) / f"{f}.pt"), map_location=dev)
    st = [b for n, b in m.named_buffers() if n.rsplit(".", 1)[-1] in STATE]
    ps = list(m.parameters()) + [b for n, b in m.named_buffers() if n.rsplit(".", 1)[-1] in ("w", "b")]
    for p in ps:
        p.requires_grad_(True)
    return m, st, ps


def step_fn(m, st, opt, X, Xs, lab, w, mask):
    opt.zero_grad(set_to_none=False)
    for b in st:
        b.detach_(); b.zero_()
    z = m(X)[:, :3]
    for b in st:
        b.detach_(); b.zero_()
    zs = m(Xs)[:, :3]
    ce = (w * F.cross_entropy(zs, lab, reduction="none")).sum() / w.sum()
    kl = (mask * (F.softmax(z.detach(), 1) * (F.log_softmax(z.detach(), 1) - F.log_softmax(zs, 1))).sum(1)).sum() / mask.sum()
    (ce + kl).backward(); opt.step()


def bench(pkg, f, dev, t, graph, n=20):
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    m, st, ps = setup(pkg, f, dev); opt = torch.optim.Adam(ps, lr=5e-4, capturable=graph)
    T = 192 if graph else t  # padded size for the graph
    X = torch.zeros(T, 41, 480, device=dev); X[:t] = torch.randn(t, 41, 480, device=dev)
    Xs = strong(X, g); lab = torch.randint(0, 3, (T,), device=dev)
    mask = torch.zeros(T, device=dev); mask[:t] = 1; w = torch.rand(T, device=dev) * mask
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    if graph:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step_fn(m, st, opt, X, Xs, lab, w, mask)
        torch.cuda.current_stream().wait_stream(s); G = torch.cuda.CUDAGraph()
        with torch.cuda.graph(G):
            step_fn(m, st, opt, X, Xs, lab, w, mask)
        run = G.replay
    else:
        for _ in range(3):
            step_fn(m, st, opt, X, Xs, lab, w, mask)
        run = lambda: step_fn(m, st, opt, X, Xs, lab, w, mask)
    sync(); t0 = time.time()
    for _ in range(n):
        run()
    sync(); return (time.time() - t0) / n


def main():
    pkg, dev = sys.argv[1], sys.argv[2]; torch.set_num_threads(1)
    print(json.dumps(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")), flush=True)
    for f in FAM:
        for t in (30, 110, 190):
            for graph in ((False, True) if dev == "cuda" else (False,)):
                try:
                    s = bench(pkg, f, dev, t, graph)
                    print(json.dumps(dict(net=f, t=t, graph=graph, sec_per_step=round(s, 5))), flush=True)
                except Exception as e:  # a failed capture is a result too
                    print(json.dumps(dict(net=f, t=t, graph=graph, error=repr(e)[:300])), flush=True)


if __name__ == "__main__":
    main()
