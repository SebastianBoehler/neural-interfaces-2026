"""REST4 (11 Nov, ledger s26): SessionBN nets trained 4-way with PROTEUS's own pre-cue rest windows as a fourth class.

Rest rows: for every Graz window, the 4 s from cue - 4.5 s to cue - 0.5 s of the same trial (data/proteus_windows_start-4.5.npz,
same row order as the official windows). They join their own session's stream for alignment and SessionBN statistics.
Only rest rows of the fold's training rows are used; the held-out sessions contribute no rest window. The net has four
outputs; test probabilities are the softmax over the first three (as every member).
Conditions: c3 (control: same code, no rest rows, 3 outputs), r1 (rest loss weight 1), r03 (rest loss weight 0.3).
Usage: python rest4.py TASK (seed = TASK // 6, arch = ("fb", "sh")[(TASK % 6) // 3], cond = CONDS[TASK % 3]) | python rest4.py smoke
"""
import sys, time
from pathlib import Path
WS = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import dg  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
from deep import param_groups  # noqa: E402

OUT = WS / "runs/diag/rest4/probs"; CONDS = ("c3", "r1", "r03"); WREST = {"c3": 0.0, "r1": 1.0, "r03": 0.3}


class Rest4(SB.SessionNet):
    nout, w_rest = 3, 0.0

    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted, head_only=False, avg=None):
        assert not (head_only or avg is not None or weighted or self.aug)
        self._prepare()
        groups = param_groups(self.net, lr, self.lld, head_only)
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.wd)
        nb = len(self._batches(rows, np.random.default_rng(0)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=epochs * nb)
        cw = torch.tensor(([1.0, 1.0, 1.0, self.w_rest])[:self.nout], device=self.dev)
        for _ in range(epochs):
            for j in self._batches(rows, rng):
                xb, yb = X[j].to(self.dev), y[j].to(self.dev)
                if self.rot:
                    xb = dg.rotate(xb, gen, self.rot)
                opt.zero_grad(); loss = F.cross_entropy(self.net(xb)[:, :self.nout], yb, weight=cw); loss.backward(); opt.step(); sched.step()


def data():
    d = protocol.load(); N = len(d["y"]); R = np.load(WS / "data/proteus_windows_start-4.5.npz")
    for k in ("y", "subject", "session", "run"):
        assert (R[k] == d[k][:len(R[k])]).all() and len(R[k]) == N, k
    sel = np.flatnonzero(np.char.endswith(d["task"].astype(str), "Graz")); d2 = {}
    for k, v in d.items():
        d2[k] = np.concatenate([v, v[sel]]) if isinstance(v, np.ndarray) and len(v) == N and k != "ch_names" else v
    d2["X"] = np.concatenate([d["X"], R["X"][sel].astype(d["X"].dtype)]); d2["y"] = np.concatenate([d["y"], np.full(len(sel), 3, d["y"].dtype)])
    return d, d2, sel


def make(arch, cond, dd, seed):
    nout = 3 if cond == "c3" else 4
    m = Rest4(dd, arch={"fb": "fbcnet", "sh": "shallow"}[arch], seed=seed, group=32, net_kw={} if nout == 3 else {"n_outputs": 4})
    m.rot = 0.05 if arch == "fb" else 0.0; m.nout, m.w_rest = nout, WREST[cond]; return m


def run(arch, cond, d, d2, sel, proto, seed, out, epochs=0):
    tag = f"rest4_{arch}_{cond}"
    if (out / f"{tag}-{proto}-s{seed}.npz").exists():
        return print("exists", tag, proto, seed, flush=True)
    N = len(d["y"]); dd = d if cond == "c3" else d2; idx_all, P_all = [], []; t0 = time.time()
    for fold, tr, te in protocol.folds(d, proto):
        tr2, te2 = (tr, te) if cond == "c3" else (np.r_[tr, tr[sel]], np.r_[te, np.zeros(len(sel), bool)])
        m = make(arch, cond, dd, seed)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr2)); m.set_k(2.0)
        idx_all.append(np.flatnonzero(te)); P_all.append(protocol.predict_streams(m, dd, te2, proba=True))
        print(f"{tag} s{seed} {proto} {fold} n_train {int(tr2.sum())} {time.time() - t0:.0f}s", flush=True)
        if epochs:
            break
    idx, P = np.concatenate(idx_all), np.concatenate(P_all); o = np.argsort(idx)
    assert np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-4)
    np.savez(out / f"{tag}-{proto}-s{seed}.npz", proba=P[o], index=idx[o])
    print(f"TOTAL {tag} s{seed} {proto} n={len(idx)} acc={np.mean(P[o].argmax(1) == d['y'][idx[o]]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d, d2, sel = data(); print("rows", len(d["y"]), "rest rows", len(sel), flush=True)
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/rest4/smoke"; out.mkdir(parents=True, exist_ok=True)
        for arch, cond in (("fb", "c3"), ("fb", "r1"), ("sh", "r03")):
            run(arch, cond, d, d2, sel, "seen_session", 0, out, epochs=1)
        return print("SMOKE OK", flush=True)
    t = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        run(("fb", "sh")[(t % 6) // 3], CONDS[t % 3], d, d2, sel, proto, t // 6, OUT)


if __name__ == "__main__":
    main()
