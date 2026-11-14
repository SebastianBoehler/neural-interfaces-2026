"""EPISODIC (14 Nov, ledger s220; ML panel 14 Nov s2.1): the fbraw SessionBN family (slot 5, fbcnet_raw_sbn, k 20)
trained inside the adaptation it is deployed with.

Arm E: every batch is one episode = Q rows per class (3Q rows) drawn from ONE training session, episodes of all sessions
in random order. SessionBN normalises the episode with its own statistics (chunk = the whole episode). The logits get a
per-class bias b (sum 0, two free parameters) solved by ITERS unrolled fixed-point steps b <- b - log mean softmax(z + b)
so that the episode's mean prediction is uniform (the chain's rebalancing does the same at test time), then
cross-entropy on z + b (the recipe's own loss, label smoothing included). Same epochs, learning rate, augmentation and
ceil(largest class count / Q) episodes per session and epoch: 49 steps per epoch on a LOSO fold against the stock 36
(1.35x; the version that ran on Galvani, job 3100517). Drop-in weights: test-time forward is the stock one.
Arm E0: the stock recipe (SessionNet, 2 chunks of 32) through this same runner: a paired control and a no-op check
(its recording-order output must match ttt7_fbraw_base of the same seed up to GPU nondeterminism).

Each fold's net is trained once and its frozen output streamed (causal SessionBN, k 20) in recording order and in the
delivery orders within 0/1, full 0/1, runs 0/1 (runs/shuffle_perms.npz, as ttt18.py). Output, shufcombo's FT_TAG format:
runs/diag/episodic/probs/ttt18<scheme><pseed><ARM>_fbraw_base-<proto>-s<seed>.npz (proba, index sorted by index).
Usage (Galvani): python -I studies/14nov/episodic/episodic.py TASK (seed = TASK % 6, arm = E if TASK < 6 else E0) | smoke"""
import importlib.util, os, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
import dg  # noqa: E402
from deep import augment, param_groups  # noqa: E402
_spec = importlib.util.spec_from_file_location("ttt5", WS / "studies/12nov/ttt5.py"); T5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T5)

FAM = "fbraw"; Q, ITERS = 21, 8  # rows per class per episode (63 rows, about the stock 64), fixed-point steps
ORDERS = [("rec", 0), ("within", 0), ("within", 1), ("full", 0), ("full", 1), ("runs", 0), ("runs", 1)]
OUT = WS / "runs/diag/episodic"


def balance_shift(z, iters=ITERS):
    """z + b with b (sum 0) such that mean softmax(z + b) over the rows is uniform; differentiable (unrolled)."""
    b = torch.zeros(1, z.shape[1], device=z.device, dtype=z.dtype)
    for _ in range(iters):
        b = b - torch.log(torch.softmax(z + b, 1).mean(0, keepdim=True))
        b = b - b.mean()
    return z + b


class EpisodicNet(SB.SessionNet):
    """SessionNet trained on single-session, class-stratified episodes with the batch-marginal logit correction."""
    resid = []  # marginal residual after the shift (smoke diagnostic)

    def _episodes(self, y, rows, rng):
        sess = self._row_sess[rows]; eps = []
        for s in np.unique(sess):
            rs = rows[sess == s]; byc = [rng.permutation(rs[y[rs] == c]) for c in range(3)]
            assert all(len(r) >= Q for r in byc), [len(r) for r in byc]
            n = int(np.ceil(max(len(r) for r in byc) / Q))  # every row once per epoch, short classes padded by resampling
            byc = [np.concatenate([r, rng.choice(r, n * Q - len(r), replace=n * Q - len(r) > len(r))]) for r in byc]
            eps += [np.concatenate([r[j * Q:(j + 1) * Q] for r in byc]) for j in range(n)]
        return [eps[i] for i in rng.permutation(len(eps))]

    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted, head_only=False, avg=None):
        assert not (head_only or avg is not None or weighted or self.dro or self.pbal or self.mstyle or self.adv)
        assert not (self.cmix or self.fsc or self.cotrain or self.adabn or self.crops), "not applied by EpisodicNet"
        self.group = 3 * Q; self._prepare()
        for mod in self.sbns():  # the whole episode is one SessionBN chunk
            mod.m = 3 * Q
        yn = y.numpy(); groups = param_groups(self.net, lr, self.lld, head_only)
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.wd)
        nb = len(self._episodes(yn, rows, np.random.default_rng(0)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=epochs * nb)
        for _ in range(epochs):
            for j in self._episodes(yn, rows, rng):
                xb, yb = X[j].to(self.dev), y[j].to(self.dev)
                if self.aug:
                    xb = augment(xb, gen)
                if self.rot:
                    xb = dg.rotate(xb, gen, self.rot)
                opt.zero_grad()
                zb = balance_shift(self.net(xb)[:, :3])
                loss = loss_fn(zb, yb)
                loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            EpisodicNet.resid.append(float((torch.softmax(zb, 1).mean(0) - 1 / 3).abs().max()))
        print(f"episodic: {nb} episodes/epoch of {3 * Q} rows, last loss {float(loss.detach()):.3f}, marginal residual {EpisodicNet.resid[-1]:.1e}", flush=True)


def make(d, seed, arm):
    m = T5.make(FAM, d, seed); assert type(m) is SB.SessionNet and not m.align and m.crops == (), (type(m), m.align)
    if arm == "E":
        m.__class__ = EpisodicNet
    return m


def ranks(proto):
    P = np.load(WS / "runs/shuffle_perms.npz"); rk = {}
    for sch, ps in ORDERS:
        if sch == "rec":
            continue
        order = P[f"{sch}|{ps}|{proto}"]; r = np.full(int(order.max()) + 10000, -1); r[order] = np.arange(len(order)); rk[sch, ps] = r
    return rk


def run(d, proto, seed, arm, out, epochs=0):
    names = {o: f"ttt18{o[0]}{o[1]}{arm}_{FAM}_base-{proto}-s{seed}.npz" for o in ORDERS}
    if all((out / "probs" / f).exists() for f in names.values()):
        return print("exists", arm, proto, seed, flush=True)
    rk, got, t0 = ranks(proto), {o: ([], []) for o in ORDERS}, time.time()
    for fold, tr, te in protocol.folds(d, proto):
        m = make(d, seed, arm)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(T5.ARCHS[FAM][1]); net = m.net.eval()
        if not epochs:
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out / "nets" / f"{arm}_{FAM}_s{seed}_{proto}_{fold}.pt")
        rows = np.flatnonzero(te); st = d["stream"][rows]; ref = protocol.predict_streams(m, d, te, proba=True); rec = []
        for s in dict.fromkeys(st):
            r0 = rows[st == s]
            for o in ORDERS:
                r = r0 if o[0] == "rec" else r0[np.argsort(rk[o][r0], kind="stable")]
                assert o[0] == "rec" or (len(set(rk[o][r])) == len(r) and (np.diff(rk[o][r]) > 0).all())
                with torch.no_grad():
                    p = torch.softmax(net(torch.from_numpy(m._aligned(r, causal=True)).to(m.dev))[:, :3].cpu().double(), 1).numpy()  # as predict_proba_stream
                got[o][0].append(p); got[o][1].append(r)
                if o[0] == "rec":
                    rec.append(p)
        dif = np.abs(np.concatenate(rec) - ref).max(); assert dif < 1e-5, dif
        print(f"{arm} s{seed} {proto} {fold} rec-vs-harness {dif:.1e} acc(rec) {np.mean(ref.argmax(1) == d['y'][rows]):.4f} {time.time() - t0:.0f}s", flush=True)
        if epochs:
            break
    for o, (ps, rs) in got.items():
        idx = np.concatenate(rs); o_ = np.argsort(idx); P = np.concatenate(ps)[o_]; assert np.isfinite(P).all() and len(set(idx)) == len(idx)
        np.savez(out / "probs" / names[o], proba=P, index=idx[o_])
        print(f"TOTAL {names[o]} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx[o_]]):.4f}", flush=True)


def main():
    d = protocol.load()
    if sys.argv[1] == "smoke":
        out = OUT / "smoke"; (out / "probs").mkdir(parents=True, exist_ok=True)
        for arm in ("E", "E0"):
            run(d, "all_loso", 0, arm, out, epochs=1)
        return print("SMOKE OK", flush=True)
    task = int(sys.argv[1]); seed, arm = task % 6, ("E", "E0")[task // 6]
    for sub in ("probs", "nets"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        run(d, proto, seed, arm, OUT)


if __name__ == "__main__":
    main()
