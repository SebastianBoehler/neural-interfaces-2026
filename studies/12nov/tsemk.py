"""TSEMK (ledger s54): the TSEM stage in k PCs of core1's scaled features (shipped: 128). Usage: python studies/12nov/tsemk.py"""
import os, sys, time, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/experiments"]
import env  # noqa: E402
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; KMAX = 1024; T0 = time.time()
CACHE = REPO + "/runs/tsemk_pcs.npz"
if os.path.exists(CACHE):
    PC = dict(np.load(CACHE))
else:
    d = protocol.load(); fe = T.Feats(d, T.parse("car_ov17_fs2_t0.1_le_k2")); st = d["stream"]; PC = {}
    for proto in A.PROTOS:
        idx = Z[f"idx|{proto}"].astype(int); pos = {int(g): k for k, g in enumerate(idx)}; out = np.zeros((len(idx), KMAX), np.float32)
        for name, tr, te in protocol.folds(d, proto):
            tri, rows = np.flatnonzero(tr), np.flatnonzero(te); sc = StandardScaler().fit(fe.F[tri]); pca = PCA(KMAX, svd_solver="full").fit(sc.transform(fe.F[tri]))
            pri = [fe.prior(b, tri) for b in range(len(fe.C))]; d2p = [q[tri].mean() for q in fe.d2]
            for s in dict.fromkeys(st[rows]):
                r = rows[st[rows] == s]; out[[pos[int(g)] for g in r]] = pca.transform(sc.transform(fe.causal(r, pri, d2p, None)))
            print(f"  PCs {proto} {name} ({time.time() - T0:.0f}s)", flush=True)
        PC[proto] = out
    np.savez_compressed(CACHE, **PC); del fe
REF = {p: Z[f"pc128|{p}"].copy() for p in A.PROTOS}
for p in A.PROTOS:
    c = [abs(np.corrcoef(REF[p][:, j], PC[p][:, j])[0, 1]) for j in range(128)]
    print(f"harness {p}: |corr| of my PCs with the stored ones, columns 0-127: min {min(c):.4f}, median {np.median(c):.4f}", flush=True)
if os.environ.get("TSEMK_SPLIT"):  # s54b: 1,024 components for new-person sessions only, fresh bases
    for bn in env.FRESH:
        for p in A.PROTOS:
            Z[f"pc128|{p}"] = REF[p]
        base = A.system(bn, A.FAMS)["out"]; Z["pc128|all_loso"] = PC["all_loso"][:, :1024].astype(np.float64); out = A.system(bn, A.FAMS)["out"]; r = SC.compare(out, base)
        print(f"SPLIT {bn}: new {r['new']:+.2f} seen {r['seen']:+.2f} | G1 {np.mean([r['dn'][SC.PEOPLE.index(q)] for q in SC.G1]):+.2f} G2 {np.mean([r['dn'][SC.PEOPLE.index(q)] for q in SC.G2]):+.2f} | new>=0 {r['npos']}/10 | "
              + " ".join(f"{q[1:]}:{v:+.1f}" for q, v in zip(SC.PEOPLE, r["dn"])), flush=True)
    for p in A.PROTOS:
        Z[f"pc128|{p}"] = REF[p]
    sys.exit(0)
for bn in env.EXPL:
    for p in A.PROTOS:
        Z[f"pc128|{p}"] = REF[p]
    A.LAM = 0.15; base = A.system(bn, A.FAMS)["out"]; print(SC.absline(bn + " shipped (stored 128 PCs)", base), flush=True)
    for k in (128, 64, 256, 512, 1024):
        for lam in ((0.15,) if k == 128 else (0.15, 0.10, 0.07)):
            for p in A.PROTOS:
                Z[f"pc128|{p}"] = PC[p][:, :k].astype(np.float64)
            A.LAM = lam; out = A.system(bn, A.FAMS)["out"]; r = SC.compare(out, base)
            print(f"   k {k:4d} lam {lam:.2f}: dJ {r['J']:+.2f} new {r['new']:+.2f} seen {r['seen']:+.2f} | G1 {r['JG1']:+.2f} G2 {r['JG2']:+.2f} | new>=0 {r['npos']}/10 ({time.time() - T0:.0f}s)", flush=True)
for p in A.PROTOS:
    Z[f"pc128|{p}"] = REF[p]
A.LAM = 0.15
