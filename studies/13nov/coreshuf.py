"""CORESHUF (ledger s175): TSEM's view (core1's 128 PCs) recomputed with core1's causal references built in a delivery
order instead of recording order (ts_core2.Feats.causal over the session's rows in delivery order; the TSEMK recipe of
s54). The recording-order rebuild is checked against the stored pc128 (median |corr| per column). Saves, per scheme,
pc128 keyed by replay row (recording index) so that shuffle.py can permute it with everything else.
Usage: [SCHEMES=within,full] [PSEED=0] python studies/13nov/coreshuf.py   -> runs/coreshuf_p<PSEED>[_<schemes>].npz"""
import os, sys, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/experiments"]
import env  # noqa: E402
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
A = env.A; Z = A.L.Z; T0 = time.time(); K = 128
SCHEMES = os.environ.get("SCHEMES", "within,full").split(","); PSEED = int(os.environ.get("PSEED", 0))
PERMS = np.load(REPO + "/runs/shuffle_perms.npz")
d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); st = d["stream"]
print(f"CORESHUF: core1 features built ({time.time() - T0:.0f}s); schemes {SCHEMES} perm {PSEED}", flush=True)
OUT = {}
for proto in A.PROTOS:
    idx = Z[f"idx|{proto}"].astype(int); pos = {int(g): k for k, g in enumerate(idx)}
    ranks = {}
    for sc in SCHEMES:
        order = PERMS[f"{sc}|{PSEED}|{proto}"]; rk = np.full(int(order.max()) + 10000, -1); rk[order] = np.arange(len(order)); ranks[sc] = rk
    out = {k: np.zeros((len(idx), K), np.float32) for k in ["rec"] + SCHEMES}
    for name, tr, te in protocol.folds(d, proto):
        tri, rows = np.flatnonzero(tr), np.flatnonzero(te); sc_ = StandardScaler().fit(fe.F[tri]); pca = PCA(K, svd_solver="full").fit(sc_.transform(fe.F[tri]))
        pri = [fe.prior(b, tri) for b in range(len(fe.C))]; d2p = [q[tri].mean() for q in fe.d2]
        for s in dict.fromkeys(st[rows]):
            r = rows[st[rows] == s]
            for key in ["rec"] + SCHEMES:
                rr = r if key == "rec" else r[np.argsort(ranks[key][r], kind="stable")]  # the session's windows in delivery order
                out[key][[pos[int(g)] for g in rr]] = pca.transform(sc_.transform(fe.causal(rr, pri, d2p, None)))
        print(f"  {proto} {name} ({time.time() - T0:.0f}s)", flush=True)
    ref = Z[f"pc128|{proto}"]; c = [abs(np.corrcoef(ref[:, j], out["rec"][:, j])[0, 1]) for j in range(K)]
    print(f"harness {proto}: rebuilt recording-order PCs against the stored ones, |corr| min {min(c):.4f} median {np.median(c):.4f}", flush=True)
    for key in SCHEMES:
        dd = np.abs(out[key] - out["rec"]).mean() / np.abs(out["rec"]).mean()
        print(f"  {proto} {key}: mean relative change of the view against recording order {dd:.3f}", flush=True)
    OUT.update({f"{key}|{proto}": v for key, v in out.items()})
np.savez_compressed(REPO + f"/runs/coreshuf_p{PSEED}{'' if SCHEMES == ['within', 'full'] else '_' + '_'.join(SCHEMES)}.npz", **OUT); print("saved", f"{time.time() - T0:.0f}s")
