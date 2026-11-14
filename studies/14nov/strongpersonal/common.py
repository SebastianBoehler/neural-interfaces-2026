"""s247 shared data: PROTEUS windows, core1's tangent features with whole-session recentring (the non-causal EA ceiling),
the routed reference (ER / T of s245, seen_session rows, 16 orders x 6 bases), the stored PFT nets, and the scorer.
Global row index = data/proteus_windows.npz order (asserted against the replay inputs)."""
import os, sys
import numpy as np

REPO = os.path.abspath(__file__ + "/../../../.."); OUT = REPO + "/runs/strongpersonal"; os.makedirs(OUT, exist_ok=True)
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402

CFG = "car_ov17_fs2_t0.1_le_k2"  # core1 (AUDIT_SBN): head lr10
D = protocol.load(); N = len(D["y"]); Y = D["y"].astype(int)
PER = np.array([s[-3:] for s in D["subject"]]); SES = D["session"].astype(str); ITF = D["interface"].astype(str)
STREAM = D["stream"]
ZI = np.load(REPO + "/runs/replay_data/inputs.npz")
SS = ZI["idx|seen_session"].astype(int)  # the 762 rows of ER / T, recording layout
assert (Y[SS] == ZI["y|seen_session"]).all() and set(PER[SS]) == {"P04", "P10"}
SEEN = ("P04", "P10")


def rows(q, s):
    return np.flatnonzero((PER == q) & (SES == s))


FOLDS = [(q, name, tr, te) for q in SEEN for name, tr, te in (("s1>s2", ["1"], "2"), ("s1>s3", ["1"], "3"), ("s12>s3", ["1", "2"], "3"))]


def ba(idx, P):
    """cell-mean balanced accuracy (%) over global rows idx (cells = person x session x interface)."""
    out = []; assert len(P) == N, "ba takes a full (N, 3) posterior"; P = P[idx]
    for c in dict.fromkeys(zip(STREAM[idx], ITF[idx])):
        m = (STREAM[idx] == c[0]) & (ITF[idx] == c[1]); yy, pp = Y[idx][m], P[m].argmax(1)
        out.append(np.mean([np.mean(pp[yy == k] == k) for k in np.unique(yy)]))
    return 100 * np.mean(out)


def ts_whole():
    """core1's features (ts_core.Feats), every session recentred on its own whole-session mean. Cached (float32)."""
    f = f"{OUT}/ts_whole.npz"
    if not os.path.exists(f):
        import ts_core
        fe = ts_core.Feats(D, ts_core.parse(CFG))
        np.savez(f, F=fe.F.astype(np.float32), slow0=np.array(fe.slices[-1].start))
    z = np.load(f); return z["F"], int(z["slow0"])


def routed(arm="ER"):
    """-> {order: [P (N, 3) per base]} with ER / T outputs on the seen_session rows (1/3 elsewhere)."""
    z = np.load(REPO + "/runs/chainout/s245_ret.npz"); out = {}
    for k in z.files:
        o, bn, a = k.split("|")
        if a == arm:
            P = np.full((N, 3), 1 / 3); P[SS] = z[k].astype(np.float64); out.setdefault(o, []).append(P)
    assert len(out) == 16 and all(len(v) == 6 for v in out.values()), {k: len(v) for k, v in out.items()}
    return out


def pft(variant="pf60", fams=("fb", "fbl", "fbraw", "sh", "sinc")):
    """geometric mean of the stored PFT nets (s29: fine-tuned on session 1 of P04 / P10), seen_session rows."""
    L = []
    for fam in fams:
        for s in range(3):
            f = f"{REPO}/runs/ttt/probs_pf/pft_{fam}_{variant}-seen_session-s{s}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f); P = np.full((N, 3), 1 / 3); P[z["index"].astype(int)] = z["proba"]; L.append(np.log(np.clip(P, 1e-4, 1)))
    G = np.exp(np.mean(L, 0)); return G / G.sum(1, keepdims=True), len(L)


def pft12(variant="pf60"):
    """geometric mean of pft12.py's nets (sessions 1 + 2 public, test session 3), if fetched to runs/strongpersonal/pft12."""
    import glob
    L = []
    for f in sorted(glob.glob(f"{OUT}/pft12/pft12_*_{variant}-s*.npz")):
        z = np.load(f); P = np.full((N, 3), 1 / 3); P[z["index"].astype(int)] = z["proba"]; L.append(np.log(np.clip(P, 1e-4, 1)))
    if not L:
        return None, 0
    G = np.exp(np.mean(L, 0)); return G / G.sum(1, keepdims=True), len(L)


def fuse(P, Q, lam):
    L = np.log(np.clip(P, 1e-6, 1)) + lam * np.log(np.clip(Q, 1e-6, 1)); L -= L.max(1, keepdims=True)
    E = np.exp(L); return E / E.sum(1, keepdims=True)
