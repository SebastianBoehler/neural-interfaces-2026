"""v6 cost (ledger §152, under §120's choices): in-session fine-tuning per never-routed session on an A100, package form.
Eager dev nets carrying the SHIPPED SessionBN weights and alignment (scripts/sbn_trace_base.py; equal to the traces),
fine-tuned on the session's aligned windows: from window 30, every 10 windows, warm start (one net copy carried
through the session, a new Adam each refit), N steps of [clean forward without grad, forward on GPU-drawn strong
augmentation, weighted CE on the chain labels + KL consistency, backward, step], then the prediction pass over the
prefix to the next refit (dev's net2(X[:hi])). Families fb (k2 traces; slot 9 = k5 stays frozen), fbraw, fbl, sinc.
Forms: all24 (six seeds per family), delta2 (two), delta1 (one); schedules cum10 (10 steps) and gC5 (5 steps).
Sessions: P01|1 (190 windows) and P04|1+2 streamed as one (378). Warm: one delta1 pass on P02|1 first.
  python studies/13nov/ft_cost_session.py ft  PKG        (needs braindecode: Leonardo .venv)
  python studies/13nov/ft_cost_session.py ref PKG        (the package itself streaming the same sessions; .venv312)
-> JSON lines with seconds per session."""
import copy, json, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402

FAM = {"fb": "sbn_k2_s{}.pt", "fbraw": "sbn_fbcnet_raw_k20_s{}.pt", "fbl": "sbn_fblight_ea_rot05_k5_s{}.pt",
       "sinc": "sbn_sincshallow_ea_ptm7r_rot05_k20_s{}.pt"}
FORMS = {"all24": 6, "delta2": 2, "delta1": 1}; SCHED = {"cum10": 10, "gC5": 5}; T0, EVERY = 30, 10


def sessions(d):
    st = d["stream"]; f = lambda s: np.flatnonzero(st == f"Dreyer2026Proteus/{s}")
    return {"warm": f("P02|1"), "190": f("P01|1"), "378": np.concatenate([f("P04|1"), f("P04|2")])}


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def strong_dev(X, g, shift=15, noise=0.2, drop=0.1):
    sh = int(torch.randint(-shift, shift + 1, (1,), generator=g, device=X.device)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g, device=X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g, device=X.device) > drop).float(); return Xs * keep


def finetune(nets, X, lab, w, steps, dev):
    """the whole session's fine-tuning for these nets (each with its own input X[i]); returns seconds and refits"""
    nets = [copy.deepcopy(n).to(dev) for n in nets]; n = len(X[0]); refits = 0
    gens = [torch.Generator(device=dev).manual_seed(s) for s in range(len(nets))]
    sync(dev); t0 = time.perf_counter()
    for t in range(T0, n, EVERY):
        refits += 1
        for net, x, g in zip(nets, X, gens):
            opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=5e-4)
            for _ in range(steps):
                opt.zero_grad()
                with torch.no_grad():
                    z = net(x[:t])[:, :3]
                zs = net(strong_dev(x[:t], g))[:, :3]
                loss = (w[:t] * F.cross_entropy(zs, lab[:t], reduction="none")).sum() / w[:t].sum() \
                    + F.kl_div(F.log_softmax(zs, 1), F.softmax(z, 1), reduction="batchmean")
                loss.backward(); opt.step()
            with torch.no_grad():
                net(x[:min(n, t + EVERY)])
    sync(dev); return time.perf_counter() - t0, refits


def main():
    mode, pkg = sys.argv[1], Path(sys.argv[2]); dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = protocol.load(); S = sessions(d); torch.set_num_threads(1)
    if mode == "ref":  # the package streaming the same sessions, one window per predict (warm)
        from add_rebalance import load_solver
        m = load_solver(pkg, pkg.name, dev)
        for tag, rows in S.items():
            c = copy.deepcopy(m); c.reset_state(); sync(dev); t0 = time.perf_counter()
            for i in rows:
                c.predict(torch.from_numpy(np.ascontiguousarray(d["X"][i:i + 1], np.float32)).to(dev))
            sync(dev)
            if tag != "warm":
                print(json.dumps(dict(mode="ref", pkg=pkg.name, session=tag, windows=len(rows), seconds=round(time.perf_counter() - t0, 1))), flush=True)
        return
    from sbn_trace_base import base_from_package
    z = np.load(pkg / "weights.npz"); idx = {str(z[f"net{i}_sjit_file"]): i for i in range(int(z["n_nets"])) if f"net{i}_sjit_file" in z.files}
    base = {(f, s): base_from_package(pkg, idx[FAM[f].format(s)], d)[0] for f in FAM for s in range(6)}
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", nets=len(base))), flush=True)
    for tag, rows in S.items():
        X = {k: torch.from_numpy(np.ascontiguousarray(m._aligned(rows, True))).to(dev) for k, m in base.items()}
        with torch.no_grad():  # chain-like labels: the frozen fb net's argmax and confidence (cost only)
            p = torch.softmax(base["fb", 0].net.to(dev)(X["fb", 0])[:, :3], 1)
        lab, w = p.argmax(1), p.max(1).values
        for sched, steps in (SCHED.items() if tag != "warm" else [("cum10", 10)]):
            for form, k in (FORMS.items() if tag != "warm" else [("delta1", 1)]):
                keys = [(f, s) for f in FAM for s in range(k)]
                sec, refits = finetune([base[q].net for q in keys], [X[q] for q in keys], lab, w, steps, dev)
                if tag != "warm":
                    print(json.dumps(dict(mode="ft", session=tag, windows=len(rows), schedule=sched, form=form, nets=len(keys),
                                          refits=refits, seconds=round(sec, 1))), flush=True)


if __name__ == "__main__":
    main()
