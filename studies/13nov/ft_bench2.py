"""v5.3 runtime, second bench (ledger s107): the s101 fine-tuning step with the parameters dev actually trains (FIR
banks are frozen buffers-in-effect: requires_grad False; SessionBN affine trained) and the clean forward without grad
(dev uses it only detached). Per family, per session length t, seconds per Adam step for
  trace1  one TorchScript trace (the shipped member; StreamBN affine buffers made trainable, state zeroed per forward)
  eager1  one eager dev net (experiments/session_bn SessionNet, eval mode = the causal formula) with the trace's weights
  vmapN   the family's N seeds in one step: torch.func.stack_module_state + functional_call + vmap, own augmentation each
Also: max |eager - trace| posterior on real windows (the eager rebuild must equal the shipped trace).
Usage (Leonardo A100): python studies/13nov/ft_bench2.py DEVICE [PKG]   -> JSON lines"""
import copy, json, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.func import functional_call, stack_module_state  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
from deep import build  # noqa: E402

FAMS = {"fb": ("fb_sbn_rot05", 2.0, "sbn_k2_s{}"), "fbraw": ("fbcnet_raw_sbn", 20.0, "sbn_fbcnet_raw_k20_s{}"),
        "fbl": ("fblight_ea_rot05_sbn", 5.0, "sbn_fblight_ea_rot05_k5_s{}"), "sinc": ("sincshallow_ea_ptm7r_rot05_sbn", 20.0, "sbn_sincshallow_ea_ptm7r_rot05_k20_s{}")}
STATE = ("s1", "s2", "n")


def strong(X, g, shift=15, noise=0.2, drop=0.1):
    sh = int(torch.randint(-shift, shift + 1, (1,), generator=g)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g).to(X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g) > drop).float().to(X.device); return Xs * keep


def eager(cfg, d, seed, k, trace, dev):
    """dev SessionNet with the shipped trace's weights (SessionBN running stats and affine from the StreamBN buffers)"""
    m = SB.make(cfg, d, seed); m.net = build(m.arch, d["X"].shape[1], d["X"].shape[2], m.pretrained, **m.net_kw); m._prepare()
    sd, ts = m.net.state_dict(), dict(trace.state_dict()); ren = {"running_mean": "rm", "running_var": "rv", "weight": "w", "bias": "b"}
    bn = {n.rsplit(".", 1)[0] for n, mod in m.net.named_modules() for n in [n + ".x"] if isinstance(mod, SB.SessionBN)}
    new = {}
    for n in sd:
        pre, last = n.rsplit(".", 1)
        new[n] = ts[f"{pre}.{ren[last]}"] if pre in bn and last in ren else ts[n]
    m.net.load_state_dict(new); m.set_k(k); net = m.net.to(dev).eval()
    for mod in net.modules():
        if isinstance(mod, SB.SessionBN):
            mod.k = k
    return net


def loss_fn(z, zs, lab, w):
    return (w * F.cross_entropy(zs, lab, reduction="none")).sum() / w.sum() + F.kl_div(F.log_softmax(zs, 1), F.softmax(z, 1), reduction="batchmean")


def timeit(fn, dev, n=10):
    for _ in range(2):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    dev = sys.argv[1]; pkg = Path(sys.argv[2] if len(sys.argv) > 2 else WS / "runs/sealed/cand_v51b_own"); torch.set_num_threads(1)
    d = protocol.load(); print(json.dumps(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")), flush=True)
    for fam, (cfg, k, pat) in FAMS.items():
        traces = [torch.jit.load(str(pkg / (pat.format(s) + ".pt")), map_location=dev) for s in range(6)]
        nets = [eager(cfg, d, s, k, tr, dev) for s, tr in enumerate(traces)]
        # eager == trace on the first 64 windows of a real session (aligned by the package's own stream is not needed for equality)
        X0 = torch.from_numpy(np.ascontiguousarray(d["X"][:64], np.float32)).to(dev) / 20.0
        with torch.no_grad():
            for tr in traces:
                for n_, b in tr.named_buffers():
                    if n_.rsplit(".", 1)[-1] in STATE:
                        b.zero_()
            diff = max(float((torch.softmax(tr(X0)[:, :3].double(), 1) - torch.softmax(nt(X0)[:, :3].double(), 1)).abs().max()) for tr, nt in zip(traces, nets))
        print(json.dumps(dict(family=fam, eager_vs_trace_max=diff)), flush=True)
        for t in (30, 110, 190):
            g = torch.Generator().manual_seed(0); X = torch.randn(t, 41, 480, device=dev)
            lab = torch.randint(0, 3, (t,), device=dev); w = torch.rand(t, device=dev)
            res = dict(family=fam, t=t)
            # trace1
            tr = copy.deepcopy(traces[0]); st = [b for n_, b in tr.named_buffers() if n_.rsplit(".", 1)[-1] in STATE]
            ps = [p for p in tr.parameters() if p.requires_grad] + [b for n_, b in tr.named_buffers() if n_.rsplit(".", 1)[-1] in ("w", "b")]
            for p in ps:
                p.requires_grad_(True)
            opt = torch.optim.Adam(ps, lr=5e-4)

            def step_trace():
                opt.zero_grad()
                for b in st:
                    b.detach_(); b.zero_()
                with torch.no_grad():
                    z = tr(X)[:, :3]
                for b in st:
                    b.detach_(); b.zero_()
                loss_fn(z, tr(strong(X, g))[:, :3], lab, w).backward(); opt.step()
            res["trace1"] = timeit(step_trace, dev)
            # eager1
            ne = copy.deepcopy(nets[0]); opt2 = torch.optim.Adam([p for p in ne.parameters() if p.requires_grad], lr=5e-4)

            def step_eager():
                opt2.zero_grad()
                with torch.no_grad():
                    z = ne(X)[:, :3]
                loss_fn(z, ne(strong(X, g))[:, :3], lab, w).backward(); opt2.step()
            res["eager1"] = timeit(step_eager, dev)
            # vmap over the six seeds
            try:
                base = copy.deepcopy(nets[0]).to("meta")
                params, bufs = stack_module_state([copy.deepcopy(n_) for n_ in nets])
                rg = {kk: v.requires_grad for kk, v in nets[0].named_parameters()}
                for kk, v in params.items():
                    v.requires_grad_(rg[kk])  # FIR banks stay frozen, as in dev
                train = {kk: v for kk, v in params.items() if rg[kk]}
                opt3 = torch.optim.Adam(list(train.values()), lr=5e-4)
                f = lambda p, b, x: functional_call(base, (p, b), (x,))[:, :3]
                gens = [torch.Generator().manual_seed(s) for s in range(6)]

                def step_vmap():
                    opt3.zero_grad()
                    with torch.no_grad():
                        z = torch.vmap(f, in_dims=(0, 0, None))(params, bufs, X)
                    Xa = torch.stack([strong(X, gg) for gg in gens])
                    zs = torch.vmap(f, in_dims=(0, 0, 0))(params, bufs, Xa)
                    sum(loss_fn(z[i], zs[i], lab, w) for i in range(6)).backward(); opt3.step()
                res["vmap6"] = timeit(step_vmap, dev)
            except Exception as e:
                res["vmap6_error"] = repr(e)[:300]
            print(json.dumps({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
