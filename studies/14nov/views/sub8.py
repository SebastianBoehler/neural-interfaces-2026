"""Sub-8 Hz view (ledger s222, step 2): per sub-8 band, m spatial components fit on the training people before the log.
  csp<m>    session-whitened windows (training: whole-session reference, test: causal, k 2), multiclass CSP = top-m
            eigenvectors of sum_k logm(M_k)^2 (M_k: training class means of the whitened covariances), view = upper(logm(U'CU))
  csp<m>n   the same without session recentring (global training reference): order-free
  pca<m>    top-m eigenvectors of the training mean covariance, m x m covariance, recentred in m dimensions (causal), tangent
  full41    the 41 x 41 tangent at the causal session reference (no reduction; the number is ignored)
  x4        4 overlapping sub-8 bands (1-3, 2-4, 3-6, 4-8) instead of 2 (1-4, 4-8)
  erp       + ERP-prototype augmented covariance on 1-8 Hz: the whitened, projected window stacked with q = 4 time courses
            (top singular vectors of the training class-average waveforms), (m + q)^2 covariance, logm
Saves runs/views/sub8.npz: <cfg>|<proto> -> member posteriors (N, 3); <cfg>|view|<proto> -> view (N, d), test rows of the
protocol (fold-specific spatial filters), NaN elsewhere; keyed by window index, recording order.
Usage: [CFGS=csp12,...] python studies/14nov/views/sub8.py"""
import os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V  # noqa: E402

T0 = time.time(); OUT = V.REPO + "/runs/views"; os.makedirs(OUT, exist_ok=True)
CFGS = os.environ.get("CFGS", "csp12,csp16,csp12n,pca12,csp12x4,csp12erp").split(",")
B2, B4, B18 = [(1, 4), (4, 8)], [(1, 3), (2, 4), (3, 6), (4, 8)], (1, 8)
need = set(B2) | (set(B4) if any("x4" in c for c in CFGS) else set()) | ({B18} if any("erp" in c for c in CFGS) else set())
XB = {b: V.band(V.XCAR, *b) for b in need}
COV = {b: V.cov(x) for b, x in XB.items()}
LOG = {b: V.logm(c) for b, c in COV.items()}
print(f"bands {sorted(need)} built ({time.time() - T0:.0f}s); cfgs {CFGS}", flush=True)
Y = V.D["y"]


def whitener(b, tr, te, recentre):
    if recentre:
        _, R = V.references(COV[b], V.session_runs(te), LOG[b][tr].mean(0))
    else:
        R = np.broadcast_to(LOG[b][tr].mean(0), COV[b].shape)
    return V.efn(R, lambda w: np.exp(-0.5 * w))


def csp(Ct, tr, m):
    S = sum(np.linalg.matrix_power(V.logm(Ct[tr & (Y == k)].mean(0)), 2) for k in range(3))
    w, U = np.linalg.eigh(S)
    return U[:, ::-1][:, :m]


def csp_block(b, tr, te, m, recentre, keep=None):
    W = whitener(b, np.flatnonzero(tr), te, recentre); Ct = np.einsum("nij,njk,nkl->nil", W, COV[b], W)
    U = csp(Ct, tr, m)
    if keep is not None:
        keep.update(W=W, U=U)
    return V.upper(V.logm(np.einsum("ji,njk,kl->nil", U, Ct, U)))


def pca_block(b, tr, te, m):
    w, U = np.linalg.eigh(V.efn(LOG[b][tr].mean(0), np.exp)); U = U[:, ::-1][:, :m]
    Cr = np.einsum("ji,njk,kl->nil", U, COV[b], U); Lr = V.logm(Cr)
    _, R = V.references(Cr, V.session_runs(te), Lr[tr].mean(0))
    return V.tangent(Cr, R)


def full_block(b, tr, te):
    _, R = V.references(COV[b], V.session_runs(te), LOG[b][tr].mean(0))
    return V.tangent(COV[b], R)


def erp_block(tr, te, m, recentre, q=4):
    kp = {}; csp_block(B18, tr, te, m, recentre, keep=kp)
    Yp = np.einsum("ji,njk,nkt->nit", kp["U"], kp["W"], XB[B18])  # whitened, projected 1-8 Hz windows (N, m, T)
    Pk = np.concatenate([Yp[tr & (Y == k)].mean(0) for k in range(3)])  # (3m, T) class-average slow waveforms
    Q = np.linalg.svd(Pk, full_matrices=False)[2][:q] * np.sqrt(Yp.shape[-1])  # unit-rms time courses (q, T)
    Za = np.concatenate([Yp, np.broadcast_to(Q, (len(Yp),) + Q.shape)], 1)
    return V.upper(V.logm(V.cov(Za)))


def feats(cfg, store):
    kind, m, rest = re.fullmatch(r"(csp|pca|full)(\d+)(\w*)", cfg).groups(); m = int(m)
    bands = B4 if "x4" in rest else B2; recentre = "n" not in rest.replace("x4", "")

    def fn(tr_idx, te):
        tr = np.zeros(V.N, bool); tr[tr_idx] = True
        blocks = [csp_block(b, tr, te, m, recentre) if kind == "csp" else pca_block(b, tr, te, m) if kind == "pca" else full_block(b, tr, te)
                  for b in bands]
        if "erp" in rest:
            blocks.append(erp_block(tr, te, m, recentre))
        F = np.concatenate(blocks, 1); store[te, :F.shape[1]] = F[te]
        return F
    return fn


if __name__ == "__main__":
    res = {}
    for cfg in CFGS:
        per = {}
        for proto in ("all_loso", "seen_session"):
            store = np.full((V.N, 4096), np.nan, np.float32)
            P = V.lopo(feats(cfg, store), protos=(proto,))[proto]
            dim = int((~np.isnan(store[np.flatnonzero(~np.isnan(P[:, 0]))[0]])).sum())
            res[f"{cfg}|{proto}"], res[f"{cfg}|view|{proto}"], per[proto] = P, store[:, :dim], P
        print(f"{cfg:10s} dim {dim:4d} {V.summary(per)} ({time.time() - T0:.0f}s)", flush=True)
    np.savez_compressed(OUT + "/sub8_" + "_".join(CFGS) + ".npz", **res)
    print("saved", f"{time.time() - T0:.0f}s")
