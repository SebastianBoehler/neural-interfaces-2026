"""NETAUG (ledger s51): SessionBN member families trained with chunk-wise "virtual person" augmentation.
One random rotation and / or frequency-scaling factor per chunk of `group` same-session windows.
Usage (Galvani): python netaug.py TASK   (0-47: arms; 100-111: in-harness twins; 120-131: cmix4; 140-151: swa50; 160-183: focal loss; 200-211: people curve; 300-335: GroupDRO / balanced; 400-423: training length) | python netaug.py smoke
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
ARMS["twin"] = dict(pw=True)  # the shipped recipe through the same class: the control (audit, ledger s52)
TWINS = [(a, "twin", s) for s in range(3) for a in ARCHS]  # tasks 100..111
ARMS["cmix4"] = dict(cmix=0.4, pw=True)  # same-class mixing across the batch, never applied to SessionBN nets (ledger s56)
CMIX = [(a, "cmix4", s) for s in range(3) for a in ARCHS]  # tasks 120..131
PCURVE = [(a, k, dr) for a in ("fb", "fbl") for k in (3, 5, 7) for dr in (0, 1)]  # tasks 200..211: people curve, seed 0
ARMS["swa50"] = dict(swa=0.5, pw=True)  # weight averaging over the second half of training, never run on SessionBN nets (ledger s57)
SWA = [(a, "swa50", s) for s in range(3) for a in ARCHS]  # tasks 140..151
ARMS["focal1"] = dict(focal=1.0, pw=True); ARMS["focal2"] = dict(focal=2.0, pw=True)  # hard-window emphasis (ledger s87)
FOCAL = [(a, f, s) for s in range(3) for f in ("focal1", "focal2") for a in ARCHS]  # tasks 160..183
ARMS["dro01"] = dict(dro=0.1, pw=True); ARMS["dro001"] = dict(dro=0.01, pw=True); ARMS["pbal"] = dict(pbal=True, pw=True)  # person-robust objectives on SessionBN nets (ledger s93)
DRO = [(a, f, s) for s in range(3) for f in ("dro01", "dro001", "pbal") for a in ARCHS]  # tasks 300..335
ARMS["ep2"] = dict(ep=2.0, pw=True); ARMS["ep05"] = dict(ep=0.5, pw=True)  # training length x2 / x0.5 (ledger s94)
EPS = [(a, f, s) for s in range(3) for f in ("ep2", "ep05") for a in ARCHS]  # tasks 400..423


class VNet(SB.SessionNet):
    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted, head_only=False, avg=None):
        assert not (head_only or avg is not None or weighted or self.dro or self.pbal or self.mstyle or self.adv)
        self._prepare(); vp, m = self.vp, self.group
        groups = param_groups(self.net, lr, self.lld, head_only); opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.wd)
        nb = len(self._batches(rows, np.random.default_rng(0)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=epochs * nb)
        step, start, mean, cnt = 0, int(vp.get("swa", 0) * epochs * nb), None, 0  # running mean of the float tensors of the state
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
                if vp.get("cmix"):
                    xb = dg.class_mix(xb, yb, gen, vp["cmix"])
                if vp.get("cfsc"):
                    xb = torch.cat([dg.freq_scale(xb[c * m:(c + 1) * m], gen, vp["cfsc"]) for c in range(G)])
                opt.zero_grad(); lo = self.net(xb)[:, :3]
                if vp.get("dro") or vp.get("pbal"):  # deep.AlignedNet._group_loss: GroupDRO over participants, or participant-balanced mean
                    old, self.dro = self.dro, vp.get("dro", 0.0)
                    try:
                        loss = self._group_loss(lo, yb, torch.from_numpy(np.asarray(j)).to(self.dev))
                    finally:
                        self.dro = old
                elif vp.get("focal"):  # focal loss: weight each window by (1 - p_true)^gamma
                    ce = torch.nn.functional.cross_entropy(lo, yb, reduction="none"); loss = ((1 - torch.exp(-ce)) ** vp["focal"] * ce).mean()
                else:
                    loss = loss_fn(lo, yb)
                loss.backward(); opt.step(); sched.step(); step += 1
                if vp.get("swa") and step > start:  # parameters and the SessionBN running statistics are averaged alike
                    sd = {k: v.detach().clone() for k, v in self.net.state_dict().items() if v.dtype.is_floating_point}; cnt += 1
                    mean = sd if mean is None else {k: mean[k] + (sd[k] - mean[k]) / cnt for k in sd}
        if vp.get("swa") and mean is not None:
            full = self.net.state_dict(); full.update(mean); self.net.load_state_dict(full); print(f"swa: averaged {cnt} steps", flush=True)


def run(arch, arm, d, proto, seed, out, epochs=0):
    cfg, k = ARCHS[arch]; f = out / f"na_{arch}_{arm}-{proto}-s{seed}.npz"
    if f.exists():
        return print("exists", f.name, flush=True)
    idx, P, t0 = [], [], time.time()
    for fold, tr, te in protocol.folds(d, proto):
        m = SB.make(cfg, d, seed); assert type(m) is SB.SessionNet; m.__class__ = VNet; m.vp = ARMS[arm]
        if ARMS[arm].get("ep"):
            m.epochs = max(1, int(round(m.epochs * ARMS[arm]["ep"]))); print(f"{arch} {arm}: epochs set to {m.epochs}", flush=True)
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


def run_pc(arch, k, draw, d, out):
    """People curve: every held-out person, trained on k random other people (shipped recipe, seed 0)."""
    cfg, kk = ARCHS[arch]; f = out / f"pc_{arch}_k{k}_d{draw}-all_loso-s0.npz"
    if f.exists():
        return print("exists", f.name, flush=True)
    per = np.array([s[-3:] for s in d["subject"]]); people = sorted(set(per)); idx, P, t0 = [], [], time.time()
    for fold, tr, te in protocol.folds(d, "all_loso"):
        held = per[np.flatnonzero(te)[0]]; rng = np.random.default_rng([people.index(held), k, draw])
        sub = rng.choice([q for q in people if q != held], k, replace=False)
        m = SB.make(cfg, d, 0); m.__class__ = VNet; m.vp = ARMS["twin"]; m.fit(np.flatnonzero(tr & np.isin(per, sub))); m.set_k(kk)
        idx.append(np.flatnonzero(te)); P.append(protocol.predict_streams(m, d, te, proba=True))
        print(f"pc {arch} k{k} d{draw} {fold} {time.time() - t0:.0f}s", flush=True)
    idx, P = np.concatenate(idx), np.concatenate(P); o = np.argsort(idx); idx, P = idx[o], P[o]
    np.savez(f, proba=P, index=idx); print(f"TOTAL {f.name} acc={np.mean(P.argmax(1) == d['y'][idx]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load()
    if sys.argv[1] == "smoke_dro":
        out = WS / "runs/diag/netaug/smoke"; out.mkdir(parents=True, exist_ok=True)
        for arm in ("dro01", "pbal"):
            run("fb", arm, d, "seen_session", 0, out, epochs=1); print("SMOKE PASS", arm, flush=True)
        return
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/netaug/smoke"; out.mkdir(parents=True, exist_ok=True)
        for arch in ARCHS:
            run(arch, "combo", d, "seen_session", 0, out, epochs=1); print("SMOKE PASS", arch, flush=True)
        return
    t = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    if t >= 300:
        arch, arm, seed = EPS[t - 400] if t >= 400 else DRO[t - 300]
        for proto in ("all_loso", "seen_session"):
            run(arch, arm, d, proto, seed, OUT)
        return
    if t >= 200:
        return run_pc(*PCURVE[t - 200], d, OUT)
    arch, arm, seed = FOCAL[t - 160] if t >= 160 else SWA[t - 140] if t >= 140 else CMIX[t - 120] if t >= 120 else TWINS[t - 100] if t >= 100 else TASKS[t]
    for proto in ("all_loso", "seen_session"):
        run(arch, arm, d, proto, seed, OUT)


if __name__ == "__main__":
    main()
