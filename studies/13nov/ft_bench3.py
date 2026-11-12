"""v5.3 runtime, third bench (ledger s107 follow-up): is the CPU-side augmentation the bottleneck? Per family and t,
seconds per Adam step for one eager net and for six seeds batched with vmap, with the augmentation drawn on the CPU
(dev's generator, then copied) or on the GPU (torch.Generator on the device), and the cost of the augmentation alone.
Usage (Leonardo A100): python studies/13nov/ft_bench3.py [PKG] -> JSON lines"""
import copy, json, sys
from pathlib import Path
import torch
from torch.func import functional_call, stack_module_state
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_bench2 as B  # noqa: E402  (eager rebuild, vmap-safe FBLight conv, loss, timer)


def strong_dev(X, g, shift=15, noise=0.2, drop=0.1):
    """the same augmentation with every draw on X's device"""
    sh = int(torch.randint(-shift, shift + 1, (1,), generator=g, device=X.device)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g, device=X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g, device=X.device) > drop).float(); return Xs * keep


def main():
    dev = "cuda"; pkg = Path(sys.argv[1] if len(sys.argv) > 1 else B.WS / "runs/sealed/cand_v51b_own"); torch.set_num_threads(1)
    d = B.protocol.load(); print(json.dumps(dict(gpu=torch.cuda.get_device_name(0))), flush=True)
    for fam, (cfg, k, pat) in B.FAMS.items():
        nets = [B.eager(cfg, d, s, k, torch.jit.load(str(pkg / (pat.format(s) + ".pt")), map_location=dev), dev) for s in range(6)]
        for t in (30, 110, 190):
            X = torch.randn(t, 41, 480, device=dev); lab = torch.randint(0, 3, (t,), device=dev); w = torch.rand(t, device=dev)
            res = dict(family=fam, t=t)
            gc, gd = torch.Generator().manual_seed(0), torch.Generator(device=dev).manual_seed(0)
            res["aug_cpu"] = B.timeit(lambda: B.strong(X, gc), dev); res["aug_gpu"] = B.timeit(lambda: strong_dev(X, gd), dev)
            for tag, aug in (("cpu", lambda: B.strong(X, gc)), ("gpu", lambda: strong_dev(X, gd))):
                ne = copy.deepcopy(nets[0]); opt = torch.optim.Adam([p for p in ne.parameters() if p.requires_grad], lr=5e-4)

                def step():
                    opt.zero_grad()
                    with torch.no_grad():
                        z = ne(X)[:, :3]
                    B.loss_fn(z, ne(aug())[:, :3], lab, w).backward(); opt.step()
                res[f"eager1_{tag}"] = B.timeit(step, dev)
            base = copy.deepcopy(nets[0]).to("meta"); params, bufs = stack_module_state([copy.deepcopy(n) for n in nets])
            rg = {kk: v.requires_grad for kk, v in nets[0].named_parameters()}
            for kk, v in params.items():
                v.requires_grad_(rg[kk])
            opt3 = torch.optim.Adam([v for kk, v in params.items() if rg[kk]], lr=5e-4)
            f = lambda p, b, x: functional_call(base, (p, b), (x,))[:, :3]; gens = [torch.Generator(device=dev).manual_seed(s) for s in range(6)]

            def step6():
                opt3.zero_grad()
                with torch.no_grad():
                    z = torch.vmap(f, in_dims=(0, 0, None))(params, bufs, X)
                zs = torch.vmap(f, in_dims=(0, 0, 0))(params, bufs, torch.stack([strong_dev(X, g) for g in gens]))
                sum(B.loss_fn(z[i], zs[i], lab, w) for i in range(6)).backward(); opt3.step()
            res["vmap6_gpu"] = B.timeit(step6, dev)
            print(json.dumps({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
