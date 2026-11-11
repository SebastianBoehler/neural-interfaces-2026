"""TTT (11 Nov, ledger s24): per-session gradient fine-tuning of a SessionBN net on the windows seen so far.

For every fold a net is trained as usual. Each test session is then streamed: every STEP windows from T0 on, a copy
of the trained net is fine-tuned for N full-batch steps on all earlier windows of the session and predicts the next
STEP windows (SessionBN statistics stay causal). Label source for the fine-tuning:
  pl  the full chain's own causal outputs for those windows (argmax, weighted by confidence)  -> packageable
  or  the true labels                                                                         -> ceiling only
Saved proba (n, 3) per variant, rows sorted by index: base (no fine-tuning), pl20, or20, or60.
Usage: python ttt.py TASK (seed = TASK // 2, arch = ("fb", "sh")[TASK % 2]) | python ttt.py smoke
"""
import copy, os, sys, time
from pathlib import Path
WS = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = WS / os.environ.get("TTT_OUT", "runs/diag/ttt/probs")
CH = np.load(HERE / os.environ.get("TTT_LABELS", "chain_final.npz")); T0, STEP, LR = 30, 10, 5e-4
VARS = {"pl20": ("pl", 20), "or20": ("or", 20), "or60": ("or", 60)}
if os.environ.get("TTT_VARS"):  # later rounds: only the chain-label variant, with the labels of the previous round's chain
    VARS = {k: v for k, v in VARS.items() if k in os.environ["TTT_VARS"].split(",")}


def make(arch, d, seed):
    if arch == "fb":
        return SB.make("fb_sbn_rot05", d, seed)
    m = SB.SessionNet(d, arch=SB.ARCH["sh"], seed=seed, **SB.CFGS["sbn"]); m.rot = 0.0; return m


def chain_labels(proto):
    if proto == "all_loso":
        return {i: CH["all_loso"][i] for i in range(len(CH["all_loso"]))}
    return {int(g): CH["seen_session"][k] for k, g in enumerate(CH["idx_seen"])}


def stream(m, X, y, Pc, seed):
    """-> {variant: (n, 3)} for one session; X aligned causal inputs (n, C, T) on the device."""
    net = m.net.eval(); n = len(X)
    with torch.no_grad():
        base = torch.softmax(net(X)[:, :3].double(), 1).cpu().numpy()
    out = {"base": base, **{v: base.copy() for v in VARS}}
    lab_pl = torch.from_numpy(Pc.argmax(1)).to(X.device); w_pl = torch.from_numpy(Pc.max(1)).float().to(X.device); lab_or = torch.from_numpy(y).to(X.device)
    for v, (src, steps) in VARS.items():
        torch.manual_seed(seed)
        for t in range(T0, n, STEP):
            net2 = copy.deepcopy(net)
            for mod in net2.modules():
                if isinstance(mod, SB.SessionBN):
                    mod.m, mod.momentum = t, 0.0
            net2.train(); opt = torch.optim.AdamW(net2.parameters(), lr=LR, weight_decay=0.0)
            lab, w = (lab_pl[:t], w_pl[:t]) if src == "pl" else (lab_or[:t], torch.ones(t, device=X.device))
            for _ in range(steps):
                opt.zero_grad(); loss = (w * F.cross_entropy(net2(X[:t])[:, :3], lab, reduction="none")).sum() / w.sum(); loss.backward(); opt.step()
            net2.eval(); hi = min(n, t + STEP)
            with torch.no_grad():
                out[v][t:hi] = torch.softmax(net2(X[:hi])[:, :3].double(), 1).cpu().numpy()[t:hi]
    return out


def run(arch, d, proto, seed, out, epochs=0):
    tag = f"ttt_{arch}"; names = ["base"] + list(VARS)
    if all((out / f"{tag}_{v}-{proto}-s{seed}.npz").exists() for v in names):
        return print("exists", tag, proto, seed, flush=True)
    got = {v: [] for v in names}; rows_all = []; t0 = time.time(); Pc_all = chain_labels(proto)
    for fold, tr, te in protocol.folds(d, proto):
        m = make(arch, d, seed)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(2.0); rows = np.flatnonzero(te); st = d["stream"][rows]
        ref = protocol.predict_streams(m, d, te, proba=True)
        for s in dict.fromkeys(st):
            r = rows[st == s]; X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev)
            res = stream(m, X, d["y"][r], np.stack([Pc_all[int(g)] for g in r]), seed)
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
        out = WS / "runs/diag/ttt/smoke"; out.mkdir(parents=True, exist_ok=True)
        for arch in ("fb", "sh"):
            run(arch, d, "seen_session", 0, out, epochs=1)
        return print("SMOKE OK", flush=True)
    task = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        run(("fb", "sh")[task % 2], d, proto, task // 2, OUT)


if __name__ == "__main__":
    main()
