"""COREBLOCKS (ledger s207, in-net edits of core1): core1 streamed as in coremember.py, saving per window the logit
contribution of each of its feature blocks (17 bands x {full, half 1, half 2} tangent blocks + the slow block) and the
intercept, in recording order and in the given delivery orders. Any block-gain edit z = sum_k g_k z_k + b of core1's
head is then exact on CPU (a gain on a block's standardised features = a gain on its head coefficients; a gain on the raw
features would be undone by the scaler). Asserts: the blocks plus intercept reproduce the head's posteriors (1e-6), and
the head reproduces the stored member in recording order (correlation >= 0.99).
Saves runs/coreblocks_<orders>.npz: <order>|<proto> -> (N, K, 3) block logits, <order>|<proto>|b -> (N, 3) intercepts.
Usage: [ORDERS=within:0] [HEAD=lr0.3] python studies/14nov/coreblocks.py"""
import os, sys, time
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/experiments", REPO + "/studies/13nov"]
import env  # noqa: E402
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
import deliver  # noqa: E402
A = env.A; T0 = time.time(); CFG = "car_ov17_fs2_t0.1_le_k2"; HEAD = os.environ.get("HEAD", "lr0.3")
ORDERS = [tuple(x.split(":")) for x in os.environ.get("ORDERS", "within:0").split(",")]
PERMS = np.load(REPO + "/runs/shuffle_perms.npz"); d = protocol.load(); fe = T.Feats(d, T.parse(CFG)); K = len(fe.slices)
print(f"COREBLOCKS: features built ({time.time() - T0:.0f}s); {K} blocks; head {HEAD}; orders {ORDERS}", flush=True)


class BlockCore(T.Core):
    def predict_proba_stream(self, idx):
        F = self.fe.causal(idx, self.priors, self.d2p, self.szp); m = self.models[0]; sc, lr = m[0], m[-1]
        Xs = sc.transform(F); Zb = np.stack([Xs[:, s] @ lr.coef_[:, s].T for s in self.fe.slices], 1)  # (n, K, 3)
        p = m.predict_proba(F); q = T.softmax(Zb.sum(1) + lr.intercept_)
        assert np.abs(p - q).max() < 1e-6, f"block logits do not reproduce the head ({np.abs(p - q).max():.2e})"
        self.store[HEAD][idx] = p; self.blocks[idx] = Zb; self.icpt[idx] = lr.intercept_
        return p


def stream(proto):
    st = {HEAD: np.full((len(d["y"]), 3), np.nan)}; holder = {}

    def make():
        c = BlockCore(fe, [HEAD], st); c.blocks, c.icpt = holder.setdefault("b", np.full((len(d["y"]), K, 3), np.nan)), holder.setdefault("i", np.full((len(d["y"]), 3), np.nan)); return c
    protocol.evaluate(make, d, proto); return st[HEAD], holder["b"], holder["i"]


OUT = {}
for proto in A.PROTOS:
    idx = A.pl.B[f"idx|{proto}"].astype(int); p, b, i = stream(proto)
    c = np.corrcoef(p[idx].ravel(), A.C1[proto].ravel())[0, 1]; print(f"harness {proto}: correlation with the stored core1 member {c:.4f}", flush=True)
    assert c >= 0.99, f"head {HEAD} does not reproduce the stored member ({c:.4f})"
    OUT[f"rec|{proto}"], OUT[f"rec|{proto}|b"], OUT[f"rec|{proto}|p"] = b[idx].astype(np.float32), i[idx], p[idx]
    for sc, sd in ORDERS:
        deliver.install(protocol, deliver.rank_from(PERMS, sc, int(sd), proto))
        try:
            p2, b2, i2 = stream(proto)
        finally:
            deliver.uninstall(protocol)
        OUT[f"{sc}{sd}|{proto}"], OUT[f"{sc}{sd}|{proto}|b"], OUT[f"{sc}{sd}|{proto}|p"] = b2[idx].astype(np.float32), i2[idx], p2[idx]
        print(f"  {proto} {sc}{sd}: mean |change| of the member against recording order {np.abs(p2[idx] - p[idx]).mean():.4f} ({time.time() - T0:.0f}s)", flush=True)
name = REPO + "/runs/coreblocks_" + "_".join(f"{a}{b}" for a, b in ORDERS) + ".npz"; np.savez_compressed(name, **OUT); print("saved", name, f"{time.time() - T0:.0f}s")
