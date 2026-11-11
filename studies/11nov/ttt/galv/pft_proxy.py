"""PFT-proxy (11 Nov, ledger s32): within-session proxy for all ten people: runs 1-3 of session 1 are the public part
over the person's later sessions with causal SessionBN statistics. seen_session folds only.
Saved proba (n, 3) per variant: base (no fine-tuning), pf20, pf60, pf200, pfh60.
Usage: python pft.py SEED (all five families) | python pft.py smoke
"""
import copy, sys, time
from pathlib import Path
import os
WS = Path(os.environ.get("NI26_WS", "/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026"))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402

OUT = WS / os.environ.get("PFT_OUT", "runs/diag/ttt/probs_pfx")
VARS = {"pf20": (20, 5e-4, "all"), "pf60": (60, 5e-4, "all"), "pfh60": (60, 2e-3, "head")}
ARCHS = {"fb": ("fb_sbn_rot05", 2.0), "sh": (None, 2.0), "fbraw": ("fbcnet_raw_sbn", 20.0), "fbl": ("fblight_ea_rot05_sbn", 5.0), "sinc": ("sincshallow_ea_ptm7r_rot05_sbn", 20.0)}


def make(arch, d, seed):
    cfg = ARCHS[arch][0]
    if cfg is not None:
        return SB.make(cfg, d, seed)
    m = SB.SessionNet(d, arch=SB.ARCH["sh"], seed=seed, **SB.CFGS["sbn"]); m.rot = 0.0; return m


def personalise(m, rows, y, steps, lr, scope, seed):
    """Copy of the trained net, fine-tuned on the person's training rows (each session = one SessionBN chunk)."""
    st = m.d["stream"][rows]; net = copy.deepcopy(m.net); torch.manual_seed(seed)
    Xs = [torch.from_numpy(m._aligned(rows[st == s], False)).to(m.dev) for s in dict.fromkeys(st)]
    ys = [torch.from_numpy(y[rows[st == s]]).to(m.dev) for s in dict.fromkeys(st)]
    for mod in net.modules():
        if isinstance(mod, SB.SessionBN):
            mod.m, mod.momentum = 0, 0.0  # m = 0: the whole batch is one chunk; running statistics stay untouched
    net.train(); params = list(net.parameters())
    if scope == "head":
        sbn = {id(p) for mod in net.modules() if isinstance(mod, SB.SessionBN) for p in mod.parameters()}
        last = [p for p in net.parameters()][-2:]
        params = [p for p in net.parameters() if id(p) in sbn] + last
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); first = None
    for _ in range(steps):
        opt.zero_grad(); loss = sum(F.cross_entropy(net(X)[:, :3], t, reduction="sum") for X, t in zip(Xs, ys)) / len(rows)
        first = float(loss) if first is None else first; loss.backward(); opt.step()
    return net.eval(), first, float(loss)


def run(arch, d, seed, out, epochs=0):
    tag = f"pfx_{arch}"; names = ["base"] + list(VARS); proto = "proxy"
    if all((out / f"{tag}_{v}-{proto}-s{seed}.npz").exists() for v in names):
        return print("exists", tag, seed, flush=True)
    got = {v: [] for v in names}; rows_all = []; t0 = time.time(); per = np.array([s[-3:] for s in d["subject"]])
    pro = d["split"] != "external"; late = (d["session"] == "1") & (d["run"].astype(int) >= 4)
    folds = [(q, pro & ~((per == q) & ((d["session"] != "1") | late)) | ~pro, pro & (per == q) & late) for q in sorted(set(per[pro]))]
    for fold, tr, te in folds:
        m = make(arch, d, seed)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(ARCHS[arch][1]); rows = np.flatnonzero(te); st = d["stream"][rows]
        who = per[rows[0]]; pr = np.flatnonzero(tr & (per == who) & (d["split"] != "external")); assert len(pr) and (per[rows] == who).all()
        nets = {"base": m.net.eval()}
        for v, (steps, lr, scope) in VARS.items():
            nets[v], l0, l1 = personalise(m, pr, d["y"], steps, lr, scope, seed)
            print(f"{tag} s{seed} {fold} {v}: {len(pr)} own rows, loss {l0:.3f} -> {l1:.3f}", flush=True)
        for s in dict.fromkeys(st):
            r = rows[st == s]; X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev)
            with torch.no_grad():
                for v in names:
                    got[v].append(torch.softmax(nets[v](X)[:, :3].float().cpu().double(), 1).numpy())
            rows_all.append(r)
        yy = d["y"][rows]; print(f"{tag} s{seed} {fold} n {len(rows)} own {len(pr)} acc " + " ".join(f"{v} {np.mean(np.concatenate(got[v][-len(dict.fromkeys(st)):]).argmax(1) == yy):.3f}" for v in names) + f" {time.time() - t0:.0f}s", flush=True)
    idx = np.concatenate(rows_all); o = np.argsort(idx)
    for v in names:
        P = np.concatenate(got[v])[o]; assert np.isfinite(P).all()
        np.savez(out / f"{tag}_{v}-{proto}-s{seed}.npz", proba=P, index=idx[o])
        print(f"TOTAL {tag}_{v} s{seed} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx[o]]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load()
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/ttt/smoke_pf"; out.mkdir(parents=True, exist_ok=True)
        run("fb", d, 0, out, epochs=1)
        return print("SMOKE OK", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); t = int(sys.argv[1])
    run(("fb", "sh", "fbraw", "fbl", "sinc")[t % 5], d, t // 5, OUT)


if __name__ == "__main__":
    main()
