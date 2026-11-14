"""s262 stage 0: profile one FineTune refit of the v6.5b package at prefixes t (default 190 378) on the given device.
Usage: python -I studies/14nov/refit_profile.py PKG [--device cuda] [--prefixes 190 378] [--smoke]
The session is an unrecognised one (as cpu_identity.py); the model is stepped over t real windows with refits switched
off (t0 = inf), then ft.refit(t) is timed. A monkeypatched copy of _refit carries synchronised stage timers
(lean/src65b is untouched). Prints JSON lines."""
import copy, io, json, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import ft_clock_restart, load_model  # noqa: E402

T = {}


def tic(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def acc(k, t0, dev):
    T[k] = T.get(k, 0.0) + (tic(dev) - t0) * 1e3


def make_refit(ft, strong):
    dev = torch.device(ft.dev)

    def _refit(self, t, lab, wt):  # copy of FineTune._refit with timers; arithmetic unchanged
        t0 = tic(dev)
        lab = torch.as_tensor(np.asarray(lab), device=dev)
        wt = torch.as_tensor(np.asarray(wt, dtype=np.float32), device=dev)
        for s, m in self.nets.items():
            X = torch.from_numpy(np.stack(self.X[s][:t])).to(dev)
            ps = self.params(m)
            for p in ps:
                p.requires_grad_(True)
            opt = torch.optim.Adam(ps, lr=self.lr)
            acc("staging", t0, dev)
            for _ in range(self.n_steps):
                t0 = tic(dev); opt.zero_grad(); self.zero(m)
                with torch.no_grad():
                    zc = m(X)[:, :3]
                acc("clean_fwd", t0, dev); t0 = tic(dev)
                self.zero(m)
                Xa = strong(X, self.gens[s])
                acc("strong_aug", t0, dev); t0 = tic(dev)
                za = m(Xa)[:, :3]
                acc("aug_fwd", t0, dev); t0 = tic(dev)
                ce = (wt * F.cross_entropy(za, lab, reduction="none")).sum() / wt.sum().clamp_min(1e-6)
                loss = ce + self.lam * F.kl_div(F.log_softmax(za, 1), F.softmax(zc, 1), reduction="batchmean")
                loss.backward(); opt.step()
                acc("bwd_adam", t0, dev)
            self.steps += self.n_steps
            t0 = tic(dev)
            with torch.no_grad():
                self.zero(m)
                m(X)
                acc("final_streambn", t0, dev); t0 = tic(dev)
                self.dtheta = max(self.dtheta, max(float((p - q).abs().max()) for p, q in zip(self.params(m), self.theta0[s])))
            acc("dtheta", t0, dev); t0 = tic(dev)
        self.refits += 1
    return _refit


def main():
    pkg = Path(sys.argv[1]); a = sys.argv
    dev = torch.device(a[a.index("--device") + 1] if "--device" in a else "cuda")
    pre = [int(x) for x in a[a.index("--prefixes") + 1:] if x.isdigit()] if "--prefixes" in a else [190, 378]
    smoke = "--smoke" in a
    d = protocol.load(); stream = d["stream"]
    ss = list(dict.fromkeys(stream))  # no dev session has 378 windows: real windows of the stream order, concatenated
    X = d["X"][np.concatenate([np.flatnonzero(stream == s) for s in ss])][:max(pre)]
    assert len(X) >= max(pre), len(X)
    m = load_model(pkg, None, "rprof", dev.type)
    for r in (m.route, getattr(m, "routeB", None)):
        if r is not None:
            r.contrast.tau = r.identity.tau = np.inf
    ft = m.ft; strong = type(ft).step.__globals__["strong"]
    print(json.dumps(dict(device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, session="concat:" + ",".join(map(str, ss[:6])),
                          n_steps=ft.n_steps, members=len(ft.member), t0=ft.t0, every=ft.every)), flush=True)
    for t in pre:
        c = copy.deepcopy(m); c.reset_state(); ft_clock_restart(c); c.ft.t0 = 10 ** 9  # stepping only, no refit
        for x in X[:t]:
            c.step(np.nan_to_num(np.asarray(x, np.float32)))
        f = c.ft; f.load(); assert len(f.lab) == t and all(len(v) == t for v in f.X.values())
        orig = type(f)._refit
        reps = 1 if smoke else 3
        wall = []
        for i in range(reps + 1):  # first call = warm-up
            g = copy.copy(f); g.nets = {s: torch.jit.load(io.BytesIO(b), map_location=dev) for s, b in f.blobs.items()}
            g.theta0 = {s: [p.detach().clone() for p in g.params(n)] for s, n in g.nets.items()}
            g.dtheta, g.refits = 0.0, 0
            t0 = tic(dev); g.refit(t); w = (tic(dev) - t0) * 1e3
            if i:
                wall.append(w)
        # stage breakdown through the patched copy (one run)
        type(f)._refit = make_refit(f, strong)
        T.clear(); g.refit(t); stages = dict(T); type(f)._refit = orig
        res = dict(prefix=t, wall_ms_median=statistics.median(wall), wall_ms_all=wall, stages_ms={k: round(v, 1) for k, v in stages.items()})
        # FIR vs full forward
        xb = torch.from_numpy(np.stack(f.X[next(iter(f.X))][:t])).to(dev); n0 = next(iter(g.nets.values()))
        with torch.no_grad(), torch.jit.optimized_execution(False):
            fir = []
            for fn in (lambda: n0.spectral_filtering(xb), lambda: n0(xb)):
                try:
                    fn(); ts = []
                    for _ in range(3 if not smoke else 1):
                        t0 = tic(dev); fn(); ts.append((tic(dev) - t0) * 1e3)
                    fir.append(statistics.median(ts))
                except Exception as e:
                    fir.append(f"ERR {type(e).__name__}: {str(e)[:80]}")
        res["fir_ms_vs_full_fwd_ms"] = fir
        # (4) member equality
        Xs = list(f.X.items()); res["X_equal_across_members"] = all(np.array_equal(Xs[0][1], v) for _, v in Xs[1:])
        outs = []
        for s, v in Xs:
            xt = torch.from_numpy(np.stack(v[:t])).to(dev)
            outs.append(strong(xt, torch.Generator(device=dev).manual_seed(f.seed)))
        res["strong_equal_across_members_same_seed"] = all(torch.equal(outs[0], o) for o in outs[1:])
        # (3) profiler
        acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        with torch.profiler.profile(activities=acts) as prof:
            g.refit(t)
            tic(dev)
        key = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
        rows = sorted(prof.key_averages(), key=lambda e: -getattr(e, "device_time_total" if dev.type == "cuda" and hasattr(e, "device_time_total") else key))[:12]
        res["top12"] = [(e.key[:60], e.count, round(getattr(e, "device_time_total", getattr(e, key, 0)) / 1e3, 1)) for e in rows]
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
