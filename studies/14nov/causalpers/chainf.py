"""CAUSAL PERSONALISATION chain inputs (ledger s253). Per order and base of the r15 chain (runs/chainout/s242r15_<order>.npz
V6Fr, recording layout = global rows): the raw posterior and the detector-gated count-filter output (rundetect/package.py
latch + J-wprior urn, exactly as studies/14nov/stack/score.py builds it), the replay person scores of both (reference for the
scorer's zero-change check), delivery positions per order, and the scoring layout (y, cell, person). Needs the replay env
(Mac). Output runs/causalpers/chain.npz.
Usage: python studies/14nov/causalpers/chainf.py"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO / "studies/14nov/rundetect"))
import numpy as np  # noqa: E402
import package as K  # noqa: E402

ORD = [f"within{i}" for i in range(12)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]


def main():
    assert (K.ROWS == np.arange(K.C.N)).all(); SC = K.C.A.SC; B = SC.B; out = {"y": B["y|all_loso"], "cell": B["cell|all_loso"], "person": B["person|all_loso"], "sess": K.C.SES}
    for o in ORD:
        z = np.load(REPO / f"runs/chainout/s242r15_{o}.npz"); lat = K.latch(o); hc = K.J.hyp_cache(o, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
        pos = np.full(K.C.N, -1); dl = []
        for s in K.SESS:
            r = K.C.order_rows(o, s, K.PROTO); pos[r] = np.arange(len(r)); dl.append(r)
        assert (pos >= 0).all(); out[f"{o}|deliv"] = np.concatenate(dl); out[f"{o}|pos"] = pos
        keys = sorted(k for k in z.files if k.endswith("|V6Fr")); assert len(keys) == 6, keys
        for i, k in enumerate(keys):
            P = z[k].astype(np.float64); F = K.gate(P, K.J.filt_joint(P, hc, "urn"), o, lat)
            assert np.allclose(F.sum(1), 1) and np.isfinite(F).all()
            out[f"{o}|{i}|raw"], out[f"{o}|{i}|filt"] = P, F
            out[f"{o}|{i}|sc_raw"], out[f"{o}|{i}|sc_filt"] = SC.person(P, "all_loso"), SC.person(F, "all_loso")
        print(o, "raw %.2f filt %.2f" % (np.mean([out[f"{o}|{i}|sc_raw"].mean() for i in range(6)]), np.mean([out[f"{o}|{i}|sc_filt"].mean() for i in range(6)])), flush=True)
    np.savez(REPO / "runs/causalpers/chain.npz", **out); print("saved runs/causalpers/chain.npz")


if __name__ == "__main__":
    main()
