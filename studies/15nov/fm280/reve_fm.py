"""s280 REVE AMENDMENT forms R1-R3 + R0 random-init control for REVE (base, 69M), through the s271 protocol; copy of fm280.py (CBraMod) with
arch reve (deep.build: 200 Hz resample, clamp 15, braindecode REVE with the s71 weights at the pinned revision, position bank from mne_data/reve_positions.json).
Forms: rv1 = F1 recipe (1e-4/1e-3, 25 ep) | rv2 = F2 recipe (lld 0.75, base 3e-4) | rv3 = F3 (last 2 blocks + head 3e-4) | rv0 = random init, F2 recipe | rv4 = R4: R1 recipe, head on the pooled mid-layer (block 11 of 22) tokens, upper blocks dropped.
Same protocol as fm280.py (LOSO ten people, causal EA k 5, inner validation person seed % 9, shuffled within0/1 delivery, base posteriors).
Silent differences vs CBraMod: REVE has no BatchNorm either (n_sbn 0, input EA only); REVE's cls_query_token and the 4D position embedding are depth -1 (embeddings) for LLRD/freeze.
Usage: python studies/15nov/fm280/reve_fm.py TASK (TASK = form_index*6 + seed*2 + half; 5 forms x 3 seeds x 2 halves = 30; rv4 = tasks 24-29) | smoke [form] | collect"""
import hashlib, os, re, sys, time
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
import deep  # noqa: E402

BLOB = "8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe"  # sha256 of brain-bzh/reve-base model.safetensors (the s71 file, md5 d2e8b1f0...)
OUT = WS / "runs/diag/reve280"; ORD = ("within0", "within1")
# form: (pretrained, lr body, lr head, layer decay, epochs, freeze all but last 2 blocks + head, per-window z-score)
FORMS = {"rv1": (True, 1e-4, 1e-3, 1.0, 25, False, False), "rv2": (True, 3e-4, 3e-4, 0.75, 25, False, False),
         "rv3": (True, 3e-4, 3e-4, 1.0, 25, True, False), "rv0": (False, 3e-4, 3e-4, 0.75, 25, False, False),
         "rv4": (True, 1e-4, 1e-3, 1.0, 25, False, False)}  # rv4 = R1 recipe, head reads the mid-layer (first 11 of 22 blocks kept, upper 11 dropped), tokens mean-pooled over channels and patches
NAMES = list(FORMS)


class MidHead(nn.Module):
    """R4 head: (batch, chan, patch, emb) mid-layer tokens -> mean over chan and patch -> LayerNorm -> Linear (named final_layer: head lr)."""
    def __init__(self, emb, n_out=3):
        super().__init__(); self.ln, self.fc = nn.LayerNorm(emb), nn.Linear(emb, n_out)

    def forward(self, x):
        return self.fc(self.ln(x.mean((1, 2))))


def mid_layer(net):
    """Cut a Resampled(REVE) after block L//2 and read it out through MidHead (in place, once)."""
    r = net.net; L = len(r.transformer.layers); assert L == 22 and not getattr(r, "_mid", False)
    r.transformer.layers = r.transformer.layers[:L // 2]; r.use_attention_pooling = False
    r.final_layer = MidHead(r.embed_dim).to(next(r.parameters()).device); r._mid = True


class FMNet(SB.SessionNet):
    def __init__(self, d, form, seed):
        pre, lrb, lrh, lld, ep, fz, zn = FORMS[form]
        super().__init__(d, group=32, arch="reve", seed=seed, pretrained=pre, epochs=ep, lr=lrb, wd=5e-2, lld=lld, k=5.0)
        self.form, self.lrh, self.freeze, self.znorm, self.val_log = form, lrh, fz, zn, []

    def _fit(self, idx):
        self.fit_subjects = set(np.unique(self.d["subject"][idx]).tolist()); super()._fit(idx)

    def _aligned(self, idx, causal, src=None):
        if not self.znorm:
            return super()._aligned(idx, causal, src)
        X = self.d["X"][idx].astype(np.float64)  # F5: per-window, per-channel z-score (no session state at all)
        return ((X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)

    def groups(self):
        named = list(self.net.named_parameters()); dep = {n: -1 if "cls_query_token" in n else deep._depth(n) for n, _ in named}
        assert all("final_layer" in n for n, _ in named if dep[n] is None), [n for n, _ in named if dep[n] is None and "final_layer" not in n]
        L = len({v for v in dep.values()} - {None, -1}); lrb, lrh, lld = self.lr, self.lrh, self.lld
        buckets = {}
        for n, p in named:
            dn = dep[n]
            if self.freeze and dn is not None and (dn == -1 or dn < L - 2):
                p.requires_grad_(False); continue
            p.requires_grad_(True)
            lr = lrh if dn is None else lrb * lld ** (L + 1 if dn == -1 else L - dn)
            buckets.setdefault(lr, []).append(p)
        return [{"params": v, "lr": lr} for lr, v in buckets.items()]

    @torch.no_grad()
    def _val_ba(self, Xv, yv):
        self.net.eval(); p = torch.cat([self.net(Xv[i:i + 128].to(self.dev))[:, :3].argmax(1).cpu() for i in range(0, len(Xv), 128)]).numpy()
        self.net.train(); return float(np.mean([np.mean(p[yv == c] == c) for c in range(3)]))

    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted, head_only=False, avg=None):
        assert not (head_only or avg is not None or weighted or getattr(self, "am", 0) or self.dro or self.pbal) and not self.aug and not self.rot
        self._prepare()
        if self.form == "rv4":
            mid_layer(self.net)
        assert self.n_sbn == 0, "REVE has no BatchNorm; per-session normalisation is the causal input EA only"
        subj = self._grp.cpu().numpy()[rows]; people = np.unique(subj); vp = people[self.seed % len(people)]
        va, rows = rows[subj == vp], rows[subj != vp]; assert len(va) > 20 and len(rows) > 0
        groups = self.groups(); opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.wd)
        nb = len(self._batches(rows, np.random.default_rng(0)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=epochs * nb)
        Xv, yv = X[va], y[va].numpy(); best, bst = -1.0, None
        for ep in range(epochs):
            for j in self._batches(rows, rng):
                assert not np.isin(j, va).any()
                xb, yb = X[j].to(self.dev), y[j].to(self.dev)
                opt.zero_grad(); loss = loss_fn(self.net(xb)[:, :3], yb); loss.backward(); opt.step(); sched.step()
            b = self._val_ba(Xv, yv); self.val_log.append(b)
            if b > best:
                best, bst = b, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(bst); self.best_ep = int(np.argmax(self.val_log)); self.val_person = str(vp)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_weights(m, form, d):
    pre = FORMS[form][0]
    if pre:  # pretrained: must differ from the pinned weights after fine-tuning, and the pinned file must be the s71 blob
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("brain-bzh/reve-base", "model.safetensors", revision=deep.REVE_REV)
        assert sha(f) == BLOB, "pretrained file is not the s71 blob"
        from safetensors.torch import load_file
        sd = load_file(f); own = {k[len("net.net."):]: v.cpu() for k, v in m.net.state_dict().items() if k.startswith("net.net.")}
        ks = [k for k in sd if k in own and not k.startswith("final_layer.")]; assert len(ks) > 50, f"only {len(ks)} pretrained keys matched"
        diff = max(float((own[k] - sd[k]).abs().max()) for k in ks)
        assert diff > 0, "fine-tuned checkpoint equals the pretrained one"
    return pre


def run_fold(form, seed, held, d, tr, te, perms):
    fn = OUT / f"{form}_s{seed}_{held}.npz"
    if fn.exists():
        return print("exists", fn.name, flush=True)
    t0 = time.time(); m = FMNet(d, form, seed); rows = np.flatnonzero(te); trr = np.flatnonzero(tr)
    if os.environ.get("SMOKE_ROWS"):  # smoke only: every n-th training row, one epoch
        trr = trr[::int(os.environ["SMOKE_ROWS"])]; m.epochs = 1
    assert d["subject"][rows[0]] not in set(d["subject"][trr])
    m.fit(trr); m.net.eval(); assert m.n_sbn == 0 and not (set(np.unique(d["subject"][te]).tolist()) & m.fit_subjects), "held-out person in the fit"
    check_weights(m, form, d)
    st = d["stream"][rows]; res = {}

    def stream_probs(order_rank):
        P = np.zeros((len(rows), 3)); pos = {int(g): i for i, g in enumerate(rows)}
        for s in dict.fromkeys(st):
            r = rows[st == s]; r = r[np.argsort(order_rank[r], kind="stable")]; Xa = m._aligned(r, causal=True); h = len(r) // 2
            if not m.znorm:  # causality / per-session reset: a prefix gives the same inputs, so order enters only through the session's own prefix
                assert np.allclose(m._aligned(r[:h], causal=True), Xa[:h], atol=1e-5), "input normalisation not causal"
            else:
                assert np.allclose(m._aligned(r[::-1], causal=True)[::-1], Xa), "z-score input depends on delivery order"
            with torch.no_grad():
                pr = torch.cat([torch.softmax(m.net(torch.from_numpy(Xa[i:i + 64]).to(m.dev))[:, :3].double(), 1).cpu() for i in range(0, len(Xa), 64)]).numpy()
            for g, p in zip(r, pr):
                P[pos[int(g)]] = p
        return P
    for o in ORD:
        res[o] = stream_probs(perms[o])
    # harness check: recording order through this code equals protocol.predict_streams (the path the other families' base outputs use)
    ref = protocol.predict_streams(m, d, te, proba=True); rec = stream_probs(np.arange(len(d["y"]) + 10000))
    assert np.abs(rec - ref).max() < 2e-3, f"stream vs harness diff {np.abs(rec - ref).max():.2e}"
    assert np.isfinite(ref).all() and all(np.isfinite(v).all() for v in res.values())
    print(f"base-vs-harness max diff {np.abs(rec - ref).max():.2e}", flush=True)
    acc = {o: float(np.mean(res[o].argmax(1) == d["y"][rows])) for o in ORD}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(fn, rows=rows, **{f"proba_{o}": res[o] for o in ORD}, ref=ref, val_log=np.array(m.val_log), best_ep=m.best_ep)
    print(f"{form} s{seed} held {held} val_person {m.val_person} best_ep {m.best_ep}/{len(m.val_log)} valBA {max(m.val_log):.3f} "
          f"acc within0 {acc['within0']:.3f} within1 {acc['within1']:.3f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load(); P = np.load(WS / "runs/shuffle_perms.npz"); perms = {}
    for o in ORD:
        order = P[f"{o[:-1]}|{o[-1]}|all_loso"]; rk = np.full(int(order.max()) + 10000, -1); rk[order] = np.arange(len(order)); perms[o] = rk
    fl = list(protocol.folds(d, "all_loso")); assert len(fl) == 10
    if sys.argv[1] == "smoke":
        os.environ.setdefault("SMOKE_ROWS", "8"); global OUT; OUT = WS / "runs/diag/reve280_smoke"; form = sys.argv[2] if len(sys.argv) > 2 else "rv1"
        h, tr, te = fl[0]; run_fold(form, 0, h, d, tr, te, perms); return print("SMOKE OK", flush=True)
    t = int(sys.argv[1]); f, rest = divmod(t, 6); seed, half = divmod(rest, 2)
    for k, (h, tr, te) in enumerate(fl):
        if k % 2 == half:
            run_fold(NAMES[f], seed, h, d, tr, te, perms)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        import reve_collect; reve_collect.main()
    else:
        main()
