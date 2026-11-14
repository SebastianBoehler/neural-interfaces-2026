"""External-population harness v1 (ledger s216): the chain's adaptation stages, run as the replay runs them (gm.M.rebal,
gm.gate, gm.em_g, gm.tsem_g from replay/gm.py, unchanged), on public corpora, leave-one-person-out.

Member (cheap stand-in for core1): CAR, 4-band log-Euclidean tangent vectors (corpus.py), training sessions centred by
their own mean, test sessions by a causal running mean in delivery order (prior 5 windows at the population mean 0),
standardised, multinomial logistic regression (C fixed a priori). Views (substitutes, PROTEUS-specific inputs absent):
  TSEM view (pc128 on PROTEUS: 128 PCs of core1's tangent features) -> 128 PCs of the member's standardised tangent
    features, PCA fitted on the training people of the fold;
  SessionEM view (GE on PROTEUS: 32 gaze + 16 ocular-event features) -> the first 48 of those PCs (no gaze part).
Orders per test session: rec (recording), full, runs, within (shuffle.py's definitions), PERMS seeds each.
Usage: [PEOPLE=n] [PERMS=3] [NJ=6] [WS=15,30,60] python harness.py CORPUS  -> runs/extharness/res_<CORPUS>.npz"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import corpus as CO  # noqa: E402
sys.path.insert(0, CO.REPO + "/replay")
import paths  # noqa: E402,F401
import gm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

C_LR, K0, NPC, NGE = 0.05, 5.0, 128, 48
SCHEME_ID = {"full": 0, "runs": 1, "within": 2}  # as studies/13nov/shuffle.py
STAGES = ("member", "rebal", "sessionem", "tsem")
PERMS = int(os.environ.get("PERMS", 3)); WS = tuple(int(w) for w in os.environ.get("WS", "15,30,60").split(","))
assert 30 in WS, "W 30 (the shipped window) is the reference and must be run"
Z = gm.L.Z; assert gm.M.L.Z is Z and gm.L.G.Z is Z, "the stage modules do not share one store"
D = {}


def order(sess, run, scheme, seed, pi):
    """rows of one person's windows (recording order inside each session) -> delivery order, sessions kept apart."""
    out = []
    for si, s in enumerate(dict.fromkeys(sess)):
        rows = np.flatnonzero(sess == s); r = np.random.default_rng([seed, pi, si, SCHEME_ID.get(scheme, 9)])
        if scheme == "rec":
            out.append(rows); continue
        if scheme == "full":
            out.append(rows[r.permutation(len(rows))]); continue
        blocks = [rows[run[rows] == u] for u in dict.fromkeys(run[rows])]
        blocks = [blocks[i] for i in r.permutation(len(blocks))] if scheme == "runs" else [b[r.permutation(len(b))] for b in blocks]
        out.append(np.concatenate(blocks))
    o = np.concatenate(out); assert np.array_equal(np.sort(o), np.arange(len(sess))), "not a permutation"
    return o


def cell_ba(y, pred):
    return 100 * np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)])


def person_ba(P, y, sess):
    return np.mean([cell_ba(y[sess == s], P[sess == s].argmax(1)) for s in dict.fromkeys(sess)])


def stages(mix, sess, ge, pc, W, check=False):
    Z["sess|ext"], Z["GE|ext"], Z["pc128|ext"] = sess, ge, pc
    b = gm.M.rebal(mix, "ext", None) if W == 30 else gm.L.rebal(mix, sess, W=W)
    if check:
        assert np.allclose(b, gm.L.rebal(mix, sess, W=30)), "W=30 via L.rebal differs from M.rebal"
    g = gm.gate(b, "ext"); e = gm.em_g(b, "ext", g); t = gm.tsem_g(e, "ext", g)
    if check:
        assert np.allclose(t, gm.pipe_g(mix, "ext")), "stages() differs from gm.pipe_g (an override is active)"
    return {"member": mix, "rebal": b, "sessionem": e, "tsem": t}


def fold(pi):
    t0 = time.time(); per, V, Vc = D["people"][pi], D["V"], D["Vc"]
    tr, te = D["person"] != per, np.flatnonzero(D["person"] == per)
    mu, sd = Vc[tr].mean(0), Vc[tr].std(0) + 1e-6; Ztr = (Vc[tr] - mu) / sd
    clf = LogisticRegression(C=C_LR, max_iter=500, tol=1e-4).fit(Ztr, D["y"][tr])
    pca = PCA(NPC, svd_solver="randomized", random_state=0).fit(Ztr)
    sess, run, y, Vt = D["session"][te], D["run"][te], D["y"][te], V[te]
    res = {}; orders = [("rec", 0)] + [(sc, s) for sc in ("full", "runs", "within") for s in range(PERMS)]
    for sc, seed in orders:
        o = order(sess, run, sc, seed, pi); so, yo, Vo = sess[o], y[o], Vt[o].astype(np.float64)
        Xc = np.empty_like(Vo)
        for s in dict.fromkeys(so):  # causal running centring in delivery order
            r = np.flatnonzero(so == s); Xc[r] = Vo[r] - np.cumsum(Vo[r], 0) / (K0 + 1 + np.arange(len(r)))[:, None]
        Zt = (Xc - mu) / sd; P = clf.predict_proba(Zt); pc = pca.transform(Zt)
        for W in [30] + [w for w in WS if w != 30]:  # W 30 first: the no-op check needs its output
            st = stages(P, so, pc[:, :NGE].copy(), pc.copy(), W, check=(sc == "rec" and W == 30))
            if W != 30:
                b30 = res.get((sc, seed, 30, "_b"))
                assert b30 is None or not np.allclose(st["rebal"], b30), f"W {W} gave the W 30 rebalancing (no-op)"
            for k in STAGES:
                res[(sc, seed, W, k)] = person_ba(st[k], yo, so)
            if W == 30:
                res[(sc, seed, 30, "_b")] = st["rebal"]
    res = {k: v for k, v in res.items() if k[3] != "_b"}
    print(f"{per}: n {len(te)} rec member {res[('rec', 0, 30, 'member')]:.1f} tsem {res[('rec', 0, 30, 'tsem')]:.1f} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return res


def main(name):
    d = CO.feats(name); D.update(d)
    key = np.char.add(d["person"], np.char.add("|", d["session"])); Vc = d["V"].astype(np.float64)
    for k in np.unique(key):  # training-side centring: each session by its own mean
        m = key == k; Vc[m] -= Vc[m].mean(0)
    D["Vc"] = Vc.astype(np.float32); D["people"] = list(dict.fromkeys(d["person"]))
    n = int(os.environ.get("PEOPLE", len(D["people"]))); nj = int(os.environ.get("NJ", 6))
    print(f"EXTHARNESS {name}: {len(D['people'])} people ({n} scored), {len(d['y'])} windows, classes {list(d['classes'])}, "
          f"perms {PERMS}, W {WS}, C {C_LR}, PCs {NPC}/{NGE}", flush=True)
    with mp.get_context("fork").Pool(nj) as pool:
        out = pool.map(fold, range(n), chunksize=1)
    keys = sorted(out[0], key=str)
    tag = os.environ.get("TAG", "")
    path = f"{CO.OUT}/res_{name}{tag}.npz"
    np.savez(path, people=np.array(D["people"][:n]), **{"|".join(map(str, k)): np.array([o[k] for o in out]) for k in keys})
    print("saved", path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
