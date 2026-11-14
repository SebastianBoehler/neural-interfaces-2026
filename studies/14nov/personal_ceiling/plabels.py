"""LABEL-FREE CEILING pseudo-labels (ledger s252). Per order: the r15 chain's V6Fr (runs/chainout/s242r15_<order>.npz),
mean over its six bases ('<order>|raw'), and the same after the detector-gated count filter (rundetect/package.py:
latch + J-wprior urn, exactly as studies/14nov/stack/score.py builds it; filtered per base, then averaged: '<order>|filt').
Recording layout (2655, 3). Output runs/personal_ceiling/plabels_r15.npz.
Usage: python studies/14nov/personal_ceiling/plabels.py"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "studies/14nov/rundetect")]
import numpy as np  # noqa: E402
import package as K  # noqa: E402
import splits as S  # noqa: E402

ORD = [f"within{i}" for i in range(12)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]


def main():
    assert (K.ROWS == np.arange(K.C.N)).all() and K.C.N == len(S.meta()["y"]); out = {}; y = K.C.Y
    for o in ORD:
        z = np.load(REPO / f"runs/chainout/s242r15_{o}.npz"); lat = K.latch(o); hc = K.J.hyp_cache(o, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
        raw, fil = [], []
        for k in sorted(k for k in z.files if k.endswith("|V6Fr")):
            P = z[k].astype(np.float64); raw.append(P); fil.append(K.gate(P, K.J.filt_joint(P, hc, "urn"), o, lat))
        assert len(raw) == 6
        out[f"{o}|raw"], out[f"{o}|filt"] = np.mean(raw, 0), np.mean(fil, 0)
        print(o, "acc raw %.3f filt %.3f" % tuple(np.mean(out[f"{o}|{f}"].argmax(1) == y) for f in ("raw", "filt")), flush=True)
    np.savez(S.OUT / "plabels_r15.npz", **out); print("saved", S.OUT / "plabels_r15.npz")


if __name__ == "__main__":
    main()
