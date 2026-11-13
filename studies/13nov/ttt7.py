"""TTT7 (13 Nov, ledger s113): runtime-feasible schedules of the s101 fine-tuning, augmentation drawn on the GPU (as a
package would draw it). Arms: fixA (dev's CPU-drawn augmentation, in-run reference), gA (fixA with GPU-drawn
augmentation), gE20 (a refit every 20 windows), gC10E20 (warm start, 10 steps, every 20), gC5 (warm start, 5 steps,
every 10), gC10T20 (warm start, 10 steps, every 10, from window 20). Saved to runs/diag/ttt/probs7 as ttt7_*.
(TTT6 docstring follows.) TTT6 (13 Nov, ledger s108): arms that may raise the gain of the s101 fine-tuning for new people, package form.
Same nets, folds, augmentation and pseudo-labels as studies/12nov/ttt5.py (imported); per variant:
  fixA   ttt5's fixA again (in-run reference: 20 steps every 10 windows, each refit from the frozen net)
  cum10  warm start: each refit continues from the previous refit's weights, 10 steps (half the cost)
  cons2  consistency (KL) weight 2
  soft   soft pseudo-labels: CE against the chain's probabilities instead of its argmax weighted by confidence
  step5  a refit every 5 windows (twice the refits)
  head   only the SessionBN affine and the output layer are trained (cheaper backward)
  view2  two augmented views per step (CE and KL averaged over both)
Saved to runs/diag/ttt/probs6 as ttt6_<arch>_<variant>-<proto>-s<seed>.npz (rows sorted by index), plus base.
Usage (Leonardo): python studies/13nov/ttt6.py TASK (arch = ORDER[TASK % 4], seed = TASK // 4) | smoke"""
import copy, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]  # not studies/12nov: its profile.py shadows the stdlib
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("ttt5", WS / "studies/12nov/ttt5.py"); T5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T5)

OUT = WS / "runs/diag/ttt/probs7"; T0 = T5.T0; TAG = "ttt7"; SMOKE_OUT = "smoke7"; DRAW = 0; CAP = 0  # ttt14/16 override; DRAW 0 and CAP 0 keep earlier runs bit-identical
# variant -> (steps per refit, refit every, lr, scope, consistency weight, label form, warm start, views, GPU aug, first window)
VARS = {"fixA": (20, 10, 5e-4, "all", 1.0, "hard", False, 1, False, 30), "gA": (20, 10, 5e-4, "all", 1.0, "hard", False, 1, True, 30),
        "gE20": (20, 20, 5e-4, "all", 1.0, "hard", False, 1, True, 30), "gC10E20": (10, 20, 5e-4, "all", 1.0, "hard", True, 1, True, 30),
        "gC5": (5, 10, 5e-4, "all", 1.0, "hard", True, 1, True, 30), "gC10T20": (10, 10, 5e-4, "all", 1.0, "hard", True, 1, True, 20)}


def strong_dev(X, g, shift=15, noise=0.2, drop=0.1):
    """ttt5.strong with every draw on X's device (a package's form)"""
    sh = int(torch.randint(-shift, shift + 1, (1,), generator=g, device=X.device)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g, device=X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g, device=X.device) > drop).float(); return Xs * keep


def params_of(net, scope):
    if scope == "head":
        sbn = {id(p) for mod in net.modules() if isinstance(mod, SB.SessionBN) for p in mod.parameters()}
        last = [p for p in net.parameters() if p.dim() > 1 and p.shape[0] >= 3][-1]
        return [p for p in net.parameters() if id(p) in sbn or p is last]
    return list(net.parameters())


def stream(m, X, Pc, seed):
    """-> {variant: (n, 3)} for one session; X aligned causal inputs (n, C, T) on the device; Pc chain outputs (n, 3)."""
    net = m.net.eval(); n = len(X)
    with torch.no_grad():
        base = torch.softmax(net(X)[:, :3].double(), 1).cpu().numpy()
    out = {"base": base, **{v: base.copy() for v in VARS}}
    lab = torch.from_numpy(Pc.argmax(1)).to(X.device); wpl = torch.from_numpy(Pc.max(1)).float().to(X.device)
    soft = torch.from_numpy(Pc).float().to(X.device)
    for v, (steps, every, lr, scope, lam, form, warm, views, gpu, t0) in VARS.items():
        ds = seed + 1000 * DRAW; torch.manual_seed(ds); net2 = None
        gen = torch.Generator(device=X.device).manual_seed(ds) if gpu else torch.Generator().manual_seed(ds)
        aug = strong_dev if gpu else T5.strong
        for t in range(t0, n, every):
            if net2 is None or not warm:
                net2 = copy.deepcopy(net).eval()
            opt = torch.optim.AdamW(params_of(net2, scope), lr=lr, weight_decay=0.0)
            for _ in range(steps):
                lo = max(0, t - CAP) if CAP else 0  # s161: train on the last CAP windows only (linear cost); predictions still use the whole prefix
                opt.zero_grad(); z = net2(X[lo:t])[:, :3]; loss = 0.0
                for _ in range(views):
                    zs = net2(aug(X[lo:t], gen))[:, :3]
                    ce = (-(soft[lo:t] * F.log_softmax(zs, 1)).sum(1).mean() if form == "soft" else
                          (wpl[lo:t] * F.cross_entropy(zs, lab[lo:t], reduction="none")).sum() / wpl[lo:t].sum().clamp_min(1e-6))
                    loss = loss + (ce + lam * F.kl_div(F.log_softmax(zs, 1), F.softmax(z.detach(), 1), reduction="batchmean")) / views
                loss.backward(); opt.step()
            hi = min(n, t + every)
            with torch.no_grad():
                out[v][t:hi] = torch.softmax(net2(X[:hi])[:, :3].double(), 1).cpu().numpy()[t:hi]
    return out


def run(arch, d, proto, seed, out, epochs=0):
    tag = f"{TAG}_{arch}"; names = ["base"] + list(VARS)
    if all((out / f"{tag}_{v}-{proto}-s{seed}.npz").exists() for v in names):
        return print("exists", tag, proto, seed, flush=True)
    got = {v: [] for v in names}; rows_all = []; t0 = time.time(); Pc_all = T5.chain_labels(proto)
    for fold, tr, te in protocol.folds(d, proto):
        m = T5.make(arch, d, seed)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(T5.ARCHS[arch][1]); rows = np.flatnonzero(te); st = d["stream"][rows]
        ref = protocol.predict_streams(m, d, te, proba=True)
        for s in dict.fromkeys(st):
            r = rows[st == s]; X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev)
            res = stream(m, X, np.stack([Pc_all[int(g)] for g in r]), seed)
            for v in names:
                got[v].append(res[v])
            rows_all.append(r)
        base = np.concatenate(got["base"][-len(dict.fromkeys(st)):])
        print(f"{tag} s{seed} {proto} {fold} base-vs-harness max diff {np.abs(base - ref).max():.2e} {time.time() - t0:.0f}s", flush=True)
        if epochs:
            break
    idx = np.concatenate(rows_all); o = np.argsort(idx)
    for v in names:
        P = np.concatenate(got[v])[o]; assert np.isfinite(P).all()
        np.savez(out / f"{tag}_{v}-{proto}-s{seed}.npz", proba=P, index=idx[o])
        print(f"TOTAL {tag}_{v} s{seed} {proto} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx[o]]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load()
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/ttt" / SMOKE_OUT; out.mkdir(parents=True, exist_ok=True)
        run("fbl", d, "seen_session", 0, out, epochs=1); return print("SMOKE OK", flush=True)
    task = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        run(T5.ORDER[task % len(T5.ORDER)], d, proto, task // len(T5.ORDER), OUT)


if __name__ == "__main__":
    main()
