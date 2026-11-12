"""NETAUG (ledger s51): SessionBN member families trained with chunk-wise "virtual person" augmentation.
One random rotation and / or frequency-scaling factor per chunk of `group` same-session windows.
Usage (Galvani): python netaug.py TASK   (task -> arch, arm, seed) | python netaug.py smoke
"""
import sys, time
from pathlib import Path
WS = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
import dg  # noqa: E402
from deep import augment, param_groups  # noqa: E402

OUT = WS / "runs/diag/netaug/probs"
ARCHS = {"fb": ("fb_sbn_rot05", 2.0), "fbraw": ("fbcnet_raw_sbn", 20.0), "fbl": ("fblight_ea_rot05_sbn", 5.0), "sinc": ("sincshallow_ea_ptm7r_rot05_sbn", 20.0)}
ARMS = {"crot05": dict(crot=0.05, pw=False), "crot10": dict(crot=0.10, pw=False), "cfsc15": dict(cfsc=0.15, pw=True), "combo": dict(crot=0.05, cfsc=0.15, pw=True)}
TASKS = [(a, r, s) for s in range(3) for a in ARCHS for r in ARMS]


class VNet(SB.SessionNet):
    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted, head_only=False, avg=None):
        assert not (head_only or avg is not None or weighted or self.dro or self.pbal or self.mstyle or self.adv)
        self._prepare(); vp, m = self.vp, self.group
        groups = param_groups(self.net, lr, self.lld, head_only); opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.wd)
        nb = len(self._batches(rows, np.random.default_rng(0)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=epochs * nb)
        for _ in range(epochs):
            for j in self._batches(rows, rng):
                xb, yb = X[j].to(self.dev), y[j].to(self.dev); G = len(xb) // m; C = xb.shape[1]
                if self.aug:
                    xb = augment(xb, gen)
                if self.rot and vp.get("pw", True):
                    xb = dg.rotate(xb, gen, self.rot)
                if vp.get("crot"):  # one rotation per chunk: a virtual person
                    A = vp["crot"] * torch.randn(G, C, C, generator=gen); Q = torch.linalg.matrix_exp(A - A.transpose(1, 2)).to(xb.device)
                    xb = (Q.repeat_interleave(m, 0)[:len(xb)] @ xb)
                if vp.get("cfsc"):
                    xb = torch.cat([dg.freq_scale(xb[c * m:(c + 1) * m], gen, vp["cfsc"]) for c in range(G)])
                opt.zero_grad(); loss = loss_fn(self.net(xb)[:, :3], yb); loss.backward(); opt.step(); sched.step()


def run(arch, arm, d, proto, seed, out, epochs=0):
    cfg, k = ARCHS[arch]; f = out / f"na_{arch}_{arm}-{proto}-s{seed}.npz"
    if f.exists():
        return print("exists", f.name, flush=True)
    idx, P, t0 = [], [], time.time()
    for fold, tr, te in protocol.folds(d, proto):
        m = SB.make(cfg, d, seed); assert type(m) is SB.SessionNet; m.__class__ = VNet; m.vp = ARMS[arm]
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(k)
        idx.append(np.flatnonzero(te)); P.append(protocol.predict_streams(m, d, te, proba=True))
        print(f"{arch} {arm} s{seed} {proto} {fold} {time.time() - t0:.0f}s", flush=True)
        if epochs:
            break
    idx, P = np.concatenate(idx), np.concatenate(P); o = np.argsort(idx); idx, P = idx[o], P[o]
    assert np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-4)
    np.savez(f, proba=P, index=idx); print(f"TOTAL {f.name} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load()
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/netaug/smoke"; out.mkdir(parents=True, exist_ok=True)
        for arch in ARCHS:
            run(arch, "combo", d, "seen_session", 0, out, epochs=1); print("SMOKE PASS", arch, flush=True)
        return
    arch, arm, seed = TASKS[int(sys.argv[1])]; OUT.mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        run(arch, arm, d, proto, seed, OUT)


if __name__ == "__main__":
    main()
