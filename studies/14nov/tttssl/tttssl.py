"""TTT-SSL (ledger s255): test-time training with a self-supervised auxiliary loss, ceiling form (non-causal).
The shipped FBLight SessionBN recipe (fblight_ea_rot05_sbn) with its output layer widened: rows 0-2 classify, rows 3.. are
the auxiliary head. Trained jointly (CE + LAM x aux) on the nine other people (all_loso folds). Forms:
  twin  the shipped recipe (3 outputs), same seed: no auxiliary head
  C     SimCLR (NT-Xent, t 0.1) on a 64-d head between two augmented views (s147's augmentation)
  P     left-right hemisphere mirror of the aligned window, 2-way, on a random half of the batch
At test time each held-out session (recording order, eval-mode SessionBN) is adapted as a whole by the arm's loss, then
the same windows are classified. Arms (loss, scope, lr, steps): see ARMS. Pseudo-labels for pl arms: PL_NPZ (make_pl.py).
Outputs runs/tttssl/probs/tttssl_<form>_<arm>-s<seed>.npz (proba (2655, 3), index), log lines DIAG per session.
Usage (Galvani): python studies/14nov/tttssl/tttssl.py TASK (form = FORMS[TASK % 3], seed = TASK // 3) | smoke"""
import copy, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
import dg  # noqa: E402
from deep import augment, param_groups  # noqa: E402

CFG, K = "fblight_ea_rot05_sbn", 5.0; OUT = WS / "runs/tttssl/probs"; LAM = 0.5; FORMS = ("twin", "C", "P")
NOUT = {"twin": 3, "C": 3 + 64, "P": 3 + 2}
PL_NPZ = Path(os.environ.get("PL_NPZ", WS / "runs/tttssl/plabels_within0.npz"))
JOINT = {"na": None, "sA10": ("ssl", "all", 1e-3, 10), "sA40": ("ssl", "all", 1e-3, 40), "sA100": ("ssl", "all", 1e-3, 100),
         "sB40": ("ssl", "bn", 5e-3, 40), "pl40": ("pl", "all", 1e-3, 40), "spl40": ("ssl+pl", "all", 1e-3, 40)}
ARMS = {"twin": {"na": None, "pl40": ("pl", "all", 1e-3, 40), "tent40": ("tent", "bn", 5e-3, 40)}, "C": JOINT, "P": JOINT}


def mirror_index(names):
    """channel permutation that swaps left and right homologues (C3 <-> C4, PO7 <-> PO8); midline (z) stays"""
    names = [str(n) for n in names]; pos = {n: i for i, n in enumerate(names)}; out = []
    for n in names:
        if n.endswith("z"):
            out.append(pos[n]); continue
        stem = n.rstrip("0123456789"); k = int(n[len(stem):]); out.append(pos[f"{stem}{k + 1 if k % 2 else k - 1}"])
    out = np.array(out); assert (out[out] == np.arange(len(out))).all() and (out != np.arange(len(out))).sum() == 36, out
    return out


def view(X, g):
    """s147's SimCLR augmentation, every draw on X's device"""
    sh = int(torch.randint(-15, 16, (1,), generator=g, device=X.device)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + 0.2 * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g, device=X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g, device=X.device) > 0.1).float()
    amp = 0.8 + 0.45 * torch.rand(Xs.shape[0], 1, 1, generator=g, device=X.device); return Xs * keep * amp


def strong(X, g, shift=15, noise=0.2, drop=0.1):
    """ttt7.strong_dev (s15's augmentation)"""
    sh = int(torch.randint(-shift, shift + 1, (1,), generator=g, device=X.device)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g, device=X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g, device=X.device) > drop).float(); return Xs * keep


def ntxent(z1, z2, t=0.1):
    z = F.normalize(torch.cat([z1, z2]), dim=1); n = len(z1); S = z @ z.T / t
    S = S.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device); return F.cross_entropy(S, tgt)


def aux_loss(net, X, form, g, mir, acc=False):
    if form == "C":
        return ntxent(net(view(X, g))[:, 3:], net(view(X, g))[:, 3:])
    assert form == "P", form
    flag = torch.rand(len(X), generator=g, device=X.device) < 0.5
    Xm = torch.where(flag[:, None, None], X[:, mir], X); z = net(Xm)[:, 3:5]; lo = F.cross_entropy(z, flag.long())
    return (lo, float((z.argmax(1) == flag.long()).float().mean())) if acc else lo


class JointNet(SB.SessionNet):
    """SessionNet whose training loss adds LAM x the auxiliary loss on rows 3.. of the output (form C or P)."""

    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted, head_only=False, avg=None):
        assert not (head_only or avg is not None or weighted or self.dro or self.pbal or self.mstyle or self.adv)
        assert not (self.cmix or self.fsc or self.cotrain or self.adabn or self.aug), "not honoured by JointNet"
        self._prepare(); groups = param_groups(self.net, lr, self.lld, head_only)
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.wd)
        nb = len(self._batches(rows, np.random.default_rng(0)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=epochs * nb)
        gd = torch.Generator(device=self.dev).manual_seed(self.seed + 7); first = last = None; mir = self.mir.to(self.dev)
        for _ in range(epochs):
            for j in self._batches(rows, rng):
                xb, yb = X[j].to(self.dev), y[j].to(self.dev)
                if self.rot:
                    xb = dg.rotate(xb, gen, self.rot)
                opt.zero_grad(); a = aux_loss(self.net, xb, self.form, gd, mir)
                loss = loss_fn(self.net(xb)[:, :3], yb) + LAM * a; loss.backward(); opt.step(); sched.step()
                first = float(a) if first is None else first; last = float(a)
        print(f"   joint {self.form}: aux loss {first:.3f} -> {last:.3f}", flush=True)


def make(form, d, seed, mir):
    m = SB.make(CFG, d, seed); assert type(m) is SB.SessionNet and not m.aug and m.init is None and not m.net_kw, vars(m).keys()
    if form != "twin":
        m.__class__ = JointNet; m.form, m.mir = form, torch.from_numpy(mir); m.net_kw = {"n_outputs": NOUT[form]}
    return m


def params(net, scope):
    if scope == "bn":
        return [p for mod in net.modules() if isinstance(mod, SB.SessionBN) for p in mod.parameters()]
    return [p for p in net.parameters() if p.requires_grad]


def adapt(net, X, form, arm, Pc, seed, mir):
    """-> (n, 3) probabilities after adapting a copy of net on the whole session X by the arm's loss"""
    loss_kind, scope, lr, steps = arm; net2 = copy.deepcopy(net).eval(); torch.manual_seed(seed)
    g = torch.Generator(device=X.device).manual_seed(seed); opt = torch.optim.AdamW(params(net2, scope), lr=lr, weight_decay=0.0)
    lab = torch.from_numpy(Pc.argmax(1)).to(X.device); wpl = torch.from_numpy(Pc.max(1)).float().to(X.device); first = last = None
    for _ in range(steps):
        opt.zero_grad(); loss = 0.0
        if "ssl" in loss_kind:
            a = aux_loss(net2, X, form, g, mir); loss = loss + a; first = float(a) if first is None else first; last = float(a)
        if "pl" in loss_kind:
            z = net2(X)[:, :3]; zs = net2(strong(X, g))[:, :3]
            ce = (wpl * F.cross_entropy(zs, lab, reduction="none")).sum() / wpl.sum().clamp_min(1e-6)
            loss = loss + ce + F.kl_div(F.log_softmax(zs, 1), F.softmax(z.detach(), 1), reduction="batchmean")
        if loss_kind == "tent":
            p = F.softmax(net2(X)[:, :3], 1); loss = -(p * torch.log(p.clamp_min(1e-12))).sum(1).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return torch.softmax(net2(X)[:, :3].double(), 1).cpu().numpy(), (first, last)


def run(form, d, seed, out, smoke=False):
    names = list(ARMS[form]); tag = f"tttssl_{form}"
    if all((out / f"{tag}_{a}-s{seed}.npz").exists() for a in names):
        return print("exists", tag, seed, flush=True)
    PL = np.load(PL_NPZ)["proba"]; assert PL.shape == (len(d["y"]), 3) and np.isfinite(PL).all()
    mir = mirror_index(d["ch_names"]) if "ch_names" in d else mirror_index(np.load(WS / "data/proteus_windows.npz")["ch_names"])
    got = {a: [] for a in names}; rows_all = []; t0 = time.time()
    for fold, tr, te in protocol.folds(d, "all_loso"):
        m = make(form, d, seed, mir)
        if smoke:
            m.epochs = 1
        m.fit(np.flatnonzero(tr)); m.set_k(K); rows = np.flatnonzero(te); st = d["stream"][rows]
        ref = protocol.predict_streams(m, d, te, proba=True); mt = torch.from_numpy(mir).to(m.dev); base = []
        for s in dict.fromkeys(st):
            r = rows[st == s]; X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev); net = m.net.eval()
            with torch.no_grad():
                P0 = torch.softmax(net(X)[:, :3].double(), 1).cpu().numpy()
            base.append(P0); diag = []
            if form == "P":
                with torch.no_grad():
                    _, macc = aux_loss(net, X, "P", torch.Generator(device=m.dev).manual_seed(seed), mt, acc=True)
                diag.append(f"mirror_acc {macc:.3f}")
            for a in names:
                if ARMS[form][a] is None:
                    got[a].append(P0); continue
                P, (f0, f1) = adapt(net, X, form, ARMS[form][a], PL[r], seed + 17, mt); got[a].append(P)
                diag.append(f"{a} acc {np.mean(P.argmax(1) == d['y'][r]):.3f}" + (f" aux {f0:.3f}->{f1:.3f}" if f0 is not None else ""))
            print(f"DIAG {tag} s{seed} {s} n={len(r)} na acc {np.mean(P0.argmax(1) == d['y'][r]):.3f} | " + " | ".join(diag), flush=True)
            rows_all.append(r)
        dd = np.abs(np.concatenate(base) - ref).max(); print(f"{tag} s{seed} {fold} base-vs-harness max diff {dd:.2e} {time.time() - t0:.0f}s", flush=True)
        assert dd < 1e-4, dd
        if smoke:
            return print("SMOKE OK", flush=True)
    idx = np.concatenate(rows_all); o = np.argsort(idx); assert (idx[o] == np.arange(len(d["y"]))).all()
    for a in names:
        P = np.concatenate(got[a])[o]; assert np.isfinite(P).all()
        np.savez(out / f"{tag}_{a}-s{seed}.npz", proba=P, index=idx[o])
        print(f"TOTAL {tag}_{a} s{seed} acc={np.mean(P.argmax(1) == d['y']):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load(); OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "smoke":
        (OUT.parent / "smoke").mkdir(exist_ok=True)
        for form in FORMS:
            run(form, d, 0, OUT.parent / "smoke", smoke=True)
        return
    task = int(sys.argv[1]); run(FORMS[task % 3], d, task // 3, OUT)


if __name__ == "__main__":
    main()
