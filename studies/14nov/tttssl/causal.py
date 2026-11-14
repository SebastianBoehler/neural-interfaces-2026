"""Causal C10 adaptation (ledger s256). The s255 wave-2 net (FBLight SessionBN, joint SimCLR head at weight 0.1, nine other
people, all_loso folds, seeds 0-5) adapted in delivery order: from window 20, every 10 windows, 5 steps of the contrastive
loss (AdamW 5e-3, wd 0, fresh optimiser per refit, weights warm-started, as s15's ttt7 schedule) on SessionBN affine
parameters only, on the windows delivered so far (< t); the windows of the next block are classified with those weights
(stream formula: SB.to_stream's StreamBN, state carried over the session; the affine parameters are copied in at each
refit). Arm A = adapted, arm B = the same net unadapted. Delivery orders: runs/shuffle_perms.npz <scheme>|<seed>|all_loso.
Per session also: seconds and ms per window of the adaptation + classification (TIME lines, with the device name).
Zero-change check (each fold, first session of the first order): streaming with refits disabled equals the eval forward.
Outputs runs/tttssl/causal/c10-s<seed>-g<group>.npz keys A|<order>, B|<order> (2655, 3) recording layout.
Usage (Galvani): python studies/14nov/tttssl/causal.py TASK (seed = TASK % 6, order group = TASK // 6 of 5) | smoke | time"""
import copy, importlib.util, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
_s = importlib.util.spec_from_file_location("tttssl2", Path(__file__).resolve().parent / "tttssl2.py")
T2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(T2); T = T2.T; SB = T.SB; protocol = T.protocol; WS = T.WS

ORDERS = [f"{s}{i}" for s, n in (("within", 12), ("full", 4), ("runs", 4)) for i in range(n)]; GROUP = 4
START, EVERY, STEPS, LR, LAMC = 20, 10, 5, 5e-3, 0.1
OUT = WS / "runs/tttssl/causal"


def sync_affine(net2, sn):
    a = {n: m for n, m in net2.named_modules() if isinstance(m, SB.SessionBN)}; b = {n: m for n, m in sn.named_modules() if isinstance(m, SB.StreamBN)}
    assert a.keys() == b.keys() and a, (a.keys(), b.keys())
    with torch.no_grad():
        for n in a:
            b[n].w.copy_(a[n].weight); b[n].b.copy_(a[n].bias)


def causal(net, X, K, seed, mir, refit=True):
    """-> (PA, PB, seconds): X one session in delivery order (aligned causally); PB unadapted eval forward, PA the causal arm"""
    dev = X.device; n = len(X)
    with torch.no_grad():
        PB = torch.softmax(net.eval()(X)[:, :3].cpu().double(), 1).numpy()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time(); net2 = copy.deepcopy(net).eval(); sn = SB.to_stream(net, K).to(dev).eval()
    g = torch.Generator(device=dev).manual_seed(seed + 17); PA = np.empty((n, 3)); bounds = [0] + list(range(START, n, EVERY)) + [n]
    bounds = sorted(set(b for b in bounds if b <= n))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if refit and lo >= START:
            opt = torch.optim.AdamW(T.params(net2, "bn"), lr=LR, weight_decay=0.0)
            for _ in range(STEPS):
                opt.zero_grad(); loss = T.aux_loss(net2, X[:lo], "C10", g, mir); loss.backward(); opt.step()
            sync_affine(net2, sn)
        with torch.no_grad():
            PA[lo:hi] = torch.softmax(sn(X[lo:hi])[:, :3].cpu().double(), 1).numpy()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return PA, PB, time.time() - t0


def delivery(perms, order, rows, st):
    """{session: rows in delivery order} for the held-out person's rows"""
    sc = order.rstrip("0123456789"); g = perms[f"{sc}|{order[len(sc):]}|all_loso"]; rk = np.full(len(g) + 1, 10 ** 9)
    assert len(g) == 2655; rk[g] = np.arange(len(g)); return {s: rows[st == s][np.argsort(rk[rows[st == s]])] for s in dict.fromkeys(st)}


def run(d, seed, orders, out, epochs=None, first_fold=False):
    perms = np.load(WS / "runs/shuffle_perms.npz"); T.LAM = T2.LAMS["C10"]; dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    mir = T.mirror_index(d["ch_names"]); res = {f"{a}|{o}": np.full((len(d["y"]), 3), np.nan) for a in "AB" for o in orders}; t00 = time.time()
    for fold, tr, te in protocol.folds(d, "all_loso"):
        m = T2.make("C10", d, seed, mir)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(T.K); rows = np.flatnonzero(te); st = d["stream"][rows]; mt = torch.from_numpy(mir).to(m.dev); net = m.net.eval()
        for oi, order in enumerate(orders):
            for s, r in delivery(perms, order, rows, st).items():
                X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev)
                if oi == 0 and s == next(iter(dict.fromkeys(st))):  # zero-change: refits off must equal the eval forward
                    P0, PB0, _ = causal(net, X, T.K, seed, mt, refit=False); dd = np.abs(P0 - PB0).max()
                    print(f"zero-change {fold} {s}: stream vs eval forward max diff {dd:.2e}", flush=True); assert dd < 1e-4, dd
                PA, PB, sec = causal(net, X, T.K, seed, mt)
                assert np.abs(PA[:START] - PB[:START]).max() < 1e-4 or len(r) <= START, "rows before the first refit must be unadapted"
                res[f"A|{order}"][r], res[f"B|{order}"][r] = PA, PB
                print(f"TIME {dev_name} s{seed} {fold} {s} {order} n={len(r)} {sec:.2f}s {1000 * sec / len(r):.1f} ms/window | acc A {np.mean(PA.argmax(1) == d['y'][r]):.3f} B {np.mean(PB.argmax(1) == d['y'][r]):.3f}", flush=True)
        print(f"{fold} done {time.time() - t00:.0f}s", flush=True)
        if first_fold:
            return print("FIRST FOLD DONE", flush=True)
    for k, v in res.items():
        assert np.isfinite(v).all() and np.allclose(v.sum(1), 1), k
    np.savez(out, **res); print(f"SAVED {out} {time.time() - t00:.0f}s", flush=True)


def main():
    d = protocol.load(); OUT.mkdir(parents=True, exist_ok=True); a = sys.argv[1]
    if a in ("smoke", "time"):  # one epoch of training: pipeline and run time only
        T.LAM = T2.LAMS["C10"]; seed = 0
        if a == "time":
            return run(d, seed, ["within0"], None, epochs=1, first_fold=True)
        return run(d, seed, ["within0", "full0"], None, epochs=1, first_fold=True)
    task = int(a); seed, grp = task % 6, task // 6; orders = ORDERS[grp * GROUP:(grp + 1) * GROUP]; out = OUT / f"c10-s{seed}-g{grp}.npz"
    if out.exists():
        return print("exists", out)
    print(f"task {task} seed {seed} orders {orders}", flush=True); run(d, seed, orders, out)


if __name__ == "__main__":
    main()
