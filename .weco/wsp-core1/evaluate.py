"""Score optimize.edit_head: apply it to every fold's head, run the full decoding chain, print the search metric.

Search metric (printed): chain balanced accuracy on unseen people of half G1, mean of two bases.
Hidden from the optimiser (written to history.jsonl only): half G2, seen people, J, standalone.
"""
import glob, importlib.util, io, json, os, sys, time, contextlib
import numpy as np
TASK = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(TASK)); t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, REPO + "/runs/archive_scratch_11nov/secop")
    import env
A = env.A; SC = A.SC
spec = importlib.util.spec_from_file_location("opt", TASK + "/optimize.py"); opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)
meta = dict(np.load(TASK + "/pack/meta.npz", allow_pickle=True))
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def rms(Z, Ws, qid):
    v = [((Z[qid == k] @ Ws[k].T) ** 2).mean() for k in np.unique(qid)]; return float(np.sqrt(np.mean(v)))


def call(W, inner, stats):
    out = np.asarray(opt.edit_head(W.copy(), inner.copy(), {k: v.copy() for k, v in stats.items()}, {k: v.copy() for k, v in meta.items()}), dtype=np.float64)
    assert out.shape == W.shape and np.isfinite(out).all(), "edit_head must return a finite (3, D) array"
    return out - out.mean(0)


try:
    G = {p: np.full((2655, 3), np.nan) for p in A.PROTOS}
    for fn in sorted(glob.glob(TASK + "/pack/*-*.npz")):
        z = np.load(fn); proto = os.path.basename(fn).split("-")[0]; stats = dict(ho=z["ho"], ins=z["ins"]); inner = z["inner"]; qid = z["qid"]
        Wn = call(z["W"], inner, stats); ks = np.unique(qid)
        base = rms(z["Zin"], {k: inner[k] for k in ks}, qid); new = rms(z["Zin"], {k: call(inner[k], inner, stats) for k in ks}, qid)
        c = base / max(new, 1e-9); G[proto][z["rows"]] = softmax(c * (z["Zc"] @ Wn.T + z["b"]))
    P = {p: G[p][IDX[p]] for p in A.PROTOS}; assert all(np.isfinite(v).all() for v in P.values())
    fams = [f for f in A.FAMS if f != "core1"]; A.EXTRA["c1e"] = P; dn, ds = [], []
    for bn in env.EXPL:
        with contextlib.redirect_stdout(io.StringIO()):
            s = A.summary(A.system(bn, fams, extra=[("c1e", 1.0)], slot="c1e"))
        dn.append(s["dn"]); ds.append(s["ds"])
    dn, ds = np.mean(dn, 0), np.mean(ds, 0); g1 = [SC.PEOPLE.index(p) for p in SC.G1]; g2 = [SC.PEOPLE.index(p) for p in SC.G2]
    sn = SC.person(P["all_loso"], "all_loso")
    rec = dict(time=time.strftime("%H:%M:%S"), g1=float(dn[g1].mean()), g2=float(dn[g2].mean()), seen=float(ds.mean()), J=float(0.5 * dn.mean() + 0.5 * ds.mean()),
               standalone_g1=float(sn[g1].mean()), standalone_g2=float(sn[g2].mean()), secs=round(time.time() - t0, 1))
    with open(TASK + "/history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"g1_chain_ba: {rec['g1']:.4f}")
    print(f"standalone_g1_ba: {rec['standalone_g1']:.4f}")
except Exception as e:  # a broken edit scores as a failure, not as a crash of the search
    print("ERROR:", repr(e)[:300]); print("g1_chain_ba: 0.0")
