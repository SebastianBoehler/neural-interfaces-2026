"""PREST (ledger s241): the rest net (studies/11nov/rest4/rest4.py, arch sh, cond r1) re-streamed with its fourth output
kept. A forward hook on the net captures the 4 logits per window; the 3-way posterior must equal the softmax over the first
three (1e-5), so the captured p_rest belongs to exactly the member the chain mixes. Each fold is trained once and streamed
in recording order and in every delivery order of ORDERS (deliver.py; causal alignment and SessionBN state follow it).
Saved to runs/diag/restprest/<order>/rest4_sh_r1-<proto>-s<seed>.npz: index, proba (N, 3), logit (N, 4), p4 (N, 4 softmax).
Usage (Leonardo): python studies/14nov/offtask/prest.py SEED | smoke   (smoke: 1 epoch, first fold, seen_session)"""
import importlib.util, os, sys, time
from pathlib import Path
import numpy as np
import torch
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
_spec = importlib.util.spec_from_file_location("rest4", WS / "studies/11nov/rest4/rest4.py"); R4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R4)
sys.path.insert(0, str(WS / "studies/13nov"))
import deliver  # noqa: E402
PERMS = np.load(WS / "runs/shuffle_perms.npz"); CAP = {}
ORDERS = ("rec", "within0", "within1", "full0", "full1", "runs0", "runs1")
_orig = R4.Rest4.predict_proba_stream


def _stream(self, idx):
    box = {}
    def grab(mod, inp, out):  # returns None, so the module's output is unchanged
        box["z"] = out.detach()
    h = self.net.register_forward_hook(grab)
    try:
        assert not getattr(self, "tta", ()), "TTA would mix shifted forwards"
        P = _orig(self, idx)
    finally:
        h.remove()
    z = box["z"].double().cpu().numpy(); assert z.shape == (len(idx), 4), z.shape
    q = np.exp(z[:, :3] - z[:, :3].max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    assert np.abs(q - P).max() < 1e-5, f"captured logits do not reproduce the posterior ({np.abs(q - P).max():.2e})"
    for i, r in enumerate(idx):
        CAP[int(r)] = z[i]
    return P


R4.Rest4.predict_proba_stream = _stream


def split(order):
    sc = order.rstrip("0123456789"); return sc, int(order[len(sc):])


def main():
    smoke = sys.argv[1] == "smoke"; seed = 0 if smoke else int(sys.argv[1]); d, d2, sel = R4.data()
    root = WS / "runs/diag/restprest" / ("smoke" if smoke else "")
    for proto in (("seen_session",) if smoke else ("all_loso", "seen_session")):
        files = {o: root / o / f"rest4_sh_r1-{proto}-s{seed}.npz" for o in ORDERS}
        if all(f.exists() for f in files.values()):
            print("exists", proto, seed, flush=True); continue
        acc = {o: ([], [], []) for o in ORDERS}; t0 = time.time()
        for fold, tr, te in R4.protocol.folds(d, proto):
            tr2, te2 = np.r_[tr, tr[sel]], np.r_[te, np.zeros(len(sel), bool)]
            m = R4.make("sh", "r1", d2, seed)
            if smoke:
                m.epochs = 1
            m.fit(np.flatnonzero(tr2)); m.set_k(2.0); ix = np.flatnonzero(te)
            for o in ORDERS:
                CAP.clear()
                if o != "rec":
                    sc, ps = split(o); deliver.install(R4.protocol, deliver.rank_from(PERMS, sc, ps, proto))
                try:
                    P = R4.protocol.predict_streams(m, d2, te2, proba=True)
                finally:
                    deliver.uninstall(R4.protocol)
                assert all(int(r) in CAP for r in ix), "windows without captured logits"
                acc[o][0].append(ix); acc[o][1].append(P); acc[o][2].append(np.stack([CAP[int(r)] for r in ix]))
            print(f"prest s{seed} {proto} {fold} n_train {int(tr2.sum())} {time.time() - t0:.0f}s", flush=True)
            if smoke:
                break
        for o in ORDERS:
            idx, P, Z = (np.concatenate(a) for a in acc[o]); k = np.argsort(idx); idx, P, Z = idx[k], P[k], Z[k]
            p4 = np.exp(Z - Z.max(1, keepdims=True)); p4 /= p4.sum(1, keepdims=True)
            files[o].parent.mkdir(parents=True, exist_ok=True); np.savez(files[o], index=idx, proba=P, logit=Z, p4=p4)
            print(f"TOTAL {o} s{seed} {proto} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx]):.4f} mean p_rest {p4[:, 3].mean():.3f}", flush=True)
    if smoke:
        print("SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()
