"""PV: shared data for the personal-view study (ledger PV1+). All public windows in all_loso layout (index = global
window index, checked), the three views, the population eye net, delivery orders, folds and the causal standardiser.
Views: gaze32 = SessionEM's 32 frontal slow-potential columns (stateless per window); GE48 = SessionEM's full view
(gaze32 + 16 event rates, causal per session in RECORDING order as stored); pc128 = TSEM's view (core1's 128 PCs in
the held-out person's LOSO basis; core1's running recentring in recording order). Population pc128 training rows come
from runs/replay_data/feats_exact/all_loso-<P>.npz (the other nine people in P's basis), checked against inputs."""
import os
import numpy as np

REPO = os.path.abspath(__file__ + "/../../../..")
OUT = REPO + "/runs/pview"
Z = np.load(REPO + "/runs/replay_data/inputs.npz")
W = np.load(REPO + "/data/proteus_windows.npz")
N = len(W["y"])
assert (Z["idx|all_loso"] == np.arange(N)).all(), "all_loso rows are not the global window index"
Y = Z["y|all_loso"].astype(int); assert (Y == W["y"]).all()
PER = np.array([s[-3:] for s in W["subject"]]); assert (PER == Z["person|all_loso"]).all()
SES, ITF, RUN = W["session"].astype(str), W["interface"].astype(str), W["run"].astype(int)
CELL, SESS = Z["cell|all_loso"], Z["sess|all_loso"]
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]
GE = Z["GE|all_loso"]; PC = Z["pc128|all_loso"]
VIEWS = {"gaze32": GE[:, :32], "GE48": GE, "pc128": PC}
_eye = [np.load(f"{REPO}/runs/ocunet/probs/oc_nomix-all_loso-s{s}.npz") for s in range(4)]
assert all((e["index"] == np.arange(N)).all() for e in _eye)
EYE = np.mean([e["proba"] for e in _eye], 0)  # population eye net (LOSO-trained), recording-order standardisation
PERMS = np.load(REPO + "/runs/shuffle_perms.npz")
ORDERS = ["rec"] + [f"within{k}" for k in range(4)] + [f"full{k}" for k in range(2)] + [f"runs{k}" for k in range(2)]


def order_rank(order):
    """global index -> delivery rank (rec: recording order)."""
    if order == "rec":
        return np.arange(N)
    sc, sd = order.rstrip("0123456789"), order[len(order.rstrip("0123456789")):]
    g = PERMS[f"{sc}|{sd}|all_loso"].astype(int); assert len(g) == N and len(set(g)) == N
    rk = np.empty(N, int); rk[g] = np.arange(N); return rk


def pop_view(view, person):
    """-> (rows, features) of the other nine people in `view`, in the coordinates `person`'s test rows use."""
    rows = np.flatnonzero(PER != person)
    if view != "pc128":
        return rows, VIEWS[view][rows]
    z = np.load(f"{REPO}/runs/replay_data/feats_exact/all_loso-{person}.npz", allow_pickle=True)
    it = z["idx_tr"].astype(int); assert set(it) == set(rows), f"feats_exact {person}: training rows differ"
    for k in [k for k in z.files if k.endswith("|idx")]:
        ix = z[k].astype(int); assert np.abs(z[k[:-4] + "|pc"][:, :128] - PC[ix]).max() < 1e-9, f"{person} {k}: pc128 mismatch"
    o = np.argsort(it); return it[o], z["pc_tr"][o, :128]


def sess_stats(F, rows_by_session):
    """per-session (mean, var) of F over the given row groups -> pooled prior (mean of means, mean of variances)."""
    ms = [F[r].mean(0) for r in rows_by_session]; vs = [F[r].var(0) for r in rows_by_session]
    return np.mean(ms, 0), np.mean(vs, 0)


def ztrain(F, groups):
    """offline per-session z-scoring of training rows (each group = one session's training rows; whole-group stats)."""
    out = np.empty_like(F, dtype=float)
    for g in groups:
        out[g] = (F[g] - F[g].mean(0)) / np.maximum(F[g].std(0), 1e-3)
    return out


def zcausal(F, prior, K=10):
    """F: one test session's rows IN DELIVERY ORDER. Window t is standardised with the stats of windows < t blended
    with the prior (mean, var) as K pseudo-windows; no statistic from window t or later."""
    m0, v0 = prior; t = np.arange(len(F))[:, None]
    S1 = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(F, 0)[:-1]]); S2 = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(F ** 2, 0)[:-1]])
    m = (K * m0 + S1) / (K + t); v = (K * (v0 + m0 ** 2) + S2) / (K + t) - m ** 2
    return (F - m) / np.sqrt(np.maximum(v, 1e-6))


def ba(rows, pred):
    """cell-mean balanced accuracy (cells = person x session x interface) over the given rows, in %."""
    out = []
    for c in np.unique(CELL[rows]):
        m = CELL[rows] == c; yy, pp = Y[rows][m], pred[m]
        out.append(np.mean([np.mean(pp[yy == k] == k) for k in np.unique(yy)]))
    return 100 * np.mean(out)


def folds(protocol):
    """-> list of (person, name, train_rows, test_rows). XS: across sessions (P04, P10); R42: per session, train four
    runs, test one Graz + one BrainHero run (3 folds); IF: per session, train one interface, test the other."""
    out = []
    for p in PEOPLE:
        sessions = sorted(set(SES[PER == p]))
        if protocol == "XS":
            if len(sessions) < 3:
                continue
            S = {s: np.flatnonzero((PER == p) & (SES == s)) for s in sessions}
            out += [(p, "s1>s2", S["1"], S["2"]), (p, "s1>s3", S["1"], S["3"]), (p, "s12>s3", np.r_[S["1"], S["2"]], S["3"])]
            continue
        for s in sessions:
            r = np.flatnonzero((PER == p) & (SES == s))
            if protocol == "R42":
                g, b = sorted(set(RUN[r][ITF[r] == "Graz"])), sorted(set(RUN[r][ITF[r] == "BrainHero"]))
                assert len(g) == 3 and len(b) == 3, (p, s, g, b)
                for k in range(3):
                    te = r[((ITF[r] == "Graz") & (RUN[r] == g[k])) | ((ITF[r] == "BrainHero") & (RUN[r] == b[k]))]
                    out.append((p, f"s{s}k{k}", np.setdiff1d(r, te), te))
            elif protocol == "IF":
                for i in ("Graz", "BrainHero"):
                    out.append((p, f"s{s}>{'BH' if i == 'Graz' else 'Graz'}", r[ITF[r] == i], r[ITF[r] != i]))
            else:
                raise ValueError(protocol)
    return out
