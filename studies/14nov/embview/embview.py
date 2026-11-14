"""EMBVIEW (ledger s234): low-dimensional embeddings of components we already have, as extra SessionEM view columns.
Sources (rows = replay rows of all_loso = window index of data/proteus_windows.npz):
  T  teacher head features (38), runs/teachfeat_<orders>.npz: 'rec|all_loso|F' and '<order>|all_loso|F' (causal EA in that order)
  F  ST-EEGFormer middle-layer embedding of the per-window input (runs/fmemb/steeg.npz 'emb_mid_raw'; order-free)
  E  eye net penultimate embedding, 9 seeds x 128 concatenated (rec: runs/ocunet/probs; order: runs/ocushuf/<order>)
Per held-out person: columns standardised on the other nine people's rows, PCA or PLS (nine people's one-hot labels),
k in {4, 8, 16}. T and F fit the map on the recording-order rows and apply it to every key; E fits it on each key's own
rows (each order has its own eye nets: ocushuf trains with the order's standardisation).
Ceiling: LOSO logistic regression on the session-standardised view (cell-mean BA, replay scorer) and the correlation of its
errors with the member mix (A.mix B2_pool, shipped families) inside person x session x run x class cells.
Writes runs/views_emb/<src>_<form><k>.npz ('rec' and '<scheme><seed>', (2655, k)) and prints the ceiling table.
Usage: python studies/14nov/embview/embview.py [T,F,E]"""
import glob, os, sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; p0 = "all_loso"; Z = A.L.Z
ORDERS = ("within0", "within1", "full0", "full1", "runs0", "runs1")
y, per, sess = Z[f"y|{p0}"].astype(int), Z[f"person|{p0}"], Z[f"sess|{p0}"]; N = len(y)
IDX = A.pl.B[f"idx|{p0}"].astype(int); assert (IDX == np.arange(N)).all(), "all_loso replay rows are not the window index"
W = np.load(f"{REPO}/data/proteus_windows.npz"); assert (W["y"][IDX] == y).all()
RUN = np.char.add(W["task"][IDX].astype(str), np.char.add("|", W["run"][IDX].astype(str)))
CELL = np.char.add(np.char.add(sess.astype(str), "|"), np.char.add(RUN, np.char.add("|", y.astype(str))))
OUT = f"{REPO}/runs/views_emb"; os.makedirs(OUT, exist_ok=True)


def load(src):
    """-> ({key: (N, d)}, fit_on_rec)"""
    if src == "T":
        f = np.load(glob.glob(f"{REPO}/runs/teachfeat_*.npz")[0]) if len(glob.glob(f"{REPO}/runs/teachfeat_*.npz")) == 1 else None
        assert f is not None, "expected exactly one runs/teachfeat_*.npz"
        return {k: f[f"{k}|{p0}|F"].astype(np.float64) for k in ("rec",) + ORDERS}, True
    if src == "F":
        f = np.load(f"{REPO}/runs/fmemb/steeg.npz"); assert (f["y"].astype(int) == y).all(), "fmemb rows are not the replay rows"
        return {"rec": f["emb_mid_raw"].astype(np.float64)}, True
    if src == "E":
        out = {}
        for k in ("rec",) + ORDERS:
            fs = sorted(glob.glob(f"{REPO}/runs/ocunet/probs/oc_nomix-{p0}-s*.npz" if k == "rec" else f"{REPO}/runs/ocushuf/{k}/oc_nomix-{p0}-s*.npz"))
            assert len(fs) == 9, (k, len(fs)); zs = [np.load(f) for f in fs]; assert all((z["index"] == IDX).all() for z in zs), k
            out[k] = np.concatenate([z["emb"] for z in zs], 1).astype(np.float64)
        return out, False
    raise SystemExit(f"unknown source {src}")


def fold_maps(X, form, k):
    """per held-out person: (mu, sd, projector) fitted on the other nine people's rows of X."""
    maps = {}
    for q in SC.PEOPLE:
        tr = per != q; mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-9); Xs = (X[tr] - mu) / sd
        if form == "pca":
            m = PCA(k, random_state=0).fit(Xs); proj = m.transform
        else:
            m = PLSRegression(k, scale=False).fit(Xs, np.eye(3)[y[tr]]); proj = m.transform
        maps[q] = (mu, sd, proj)
    return maps


def apply(maps, X):
    out = None
    for q, (mu, sd, proj) in maps.items():
        te = per == q; v = proj((X[te] - mu) / sd)
        out = np.zeros((N, v.shape[1])) if out is None else out; out[te] = v
    return out


def sess_std(V):
    out = V.copy()
    for s in np.unique(sess):
        r = sess == s; out[r] = (V[r] - V[r].mean(0)) / np.maximum(V[r].std(0), 1e-9)
    return out


def probe(V):
    """LOSO logistic regression on the session-standardised view -> posteriors (N, 3)."""
    Vs = sess_std(V); P = np.zeros((N, 3))
    for q in SC.PEOPLE:
        te = per == q; P[te] = LogisticRegression(C=1.0, max_iter=3000).fit(Vs[~te], y[~te]).predict_proba(Vs[te])
    return P


def within(x):
    out = x.astype(np.float64).copy()
    for c in np.unique(CELL):
        r = CELL == c; out[r] -= out[r].mean()
    return out


MIX = A.mix("B2_pool", p0, A.FAMS)[0]; LPM = within(np.log(np.clip(MIX[np.arange(N), y], 1e-6, 1))); EM = within(MIX.argmax(1) != y)


def errcorr(P):
    lp = within(np.log(np.clip(P[np.arange(N), y], 1e-6, 1))); e = within(P.argmax(1) != y)
    return np.corrcoef(lp, LPM)[0, 1], np.corrcoef(e, EM)[0, 1]


def line(name, P):
    acc = SC.person(P, p0); c1, c2 = errcorr(P)
    return f"{name:14s} probe BA {acc.mean():5.1f} (people {' '.join(f'{v:.0f}' for v in acc)}) | err corr logp {c1:+.3f} err01 {c2:+.3f}"


if __name__ == "__main__":
    srcs = (sys.argv[1] if len(sys.argv) > 1 else "T,F,E").split(",")
    print(line("member mix", MIX), flush=True)
    OCU = np.mean([np.load(f)["proba"] for f in sorted(glob.glob(f"{REPO}/runs/ocunet/probs/oc_nomix-{p0}-s*.npz"))], 0)
    print(line("eye posterior", OCU) + "  (reference: the eye net's 9-seed posterior itself)", flush=True)
    print(line("eye logp probe", probe(np.log(np.clip(OCU, 1e-3, 1)))), flush=True)
    for src in srcs:
        X, on_rec = load(src)
        for form in ("pca", "pls"):
            for k in (4, 8, 16):
                views = {}
                if on_rec:
                    maps = fold_maps(X["rec"], form, k); views = {key: apply(maps, Xk) for key, Xk in X.items()}
                else:
                    views = {key: apply(fold_maps(Xk, form, k), Xk) for key, Xk in X.items()}
                name = f"{src}_{form}{k}"
                np.savez(f"{OUT}/{name}.npz", **{key: v.astype(np.float32) for key, v in views.items()})
                print(line(name, probe(views["rec"])) + (f" | within0 view {SC.person(probe(views['within0']), p0).mean():.1f}" if "within0" in views else ""), flush=True)
