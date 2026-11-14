"""CAUSAL PERSONALISATION, fusion before the count filter (ledger s253, secondary). out = filter(raw x personal^w), the
detector-gated J-wprior urn of rundetect/package.py run on the fused posterior (so the urn counts see the personal model),
against r15 + count filter. Small grid (hard labels, lE, k 10, R 80, t0 40 / 60, cap 0.15 / 0.25 / 0.5), nested choice as in
score.py. Needs the replay env (Mac).
Usage: python studies/14nov/causalpers/prefilt.py > evidence/causalpers_14nov/prefilt.txt"""
import itertools, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path[:0] = [str(HERE), str(REPO / "studies/14nov/rundetect")]
import importlib.util  # noqa: E402
import numpy as np  # noqa: E402
import package as K  # noqa: E402
_sp = importlib.util.spec_from_file_location("cpers_score", HERE / "score.py"); SC = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(SC)  # replay/ also has a score.py

GRID = list(itertools.product(("hard",), ("lE",), (10,), (40, 60), (80,), (0.15, 0.25, 0.5)))


def main():
    SC.GRID[:] = GRID; av = SC.available(); orders = list(dict.fromkeys(o for o, _ in av)); Z = SC.Z
    G = np.zeros((len(GRID), len(av), 10)); R1 = np.zeros((len(av), 10)); j = 0
    for o in orders:
        lat = K.latch(o); hc = K.J.hyp_cache(o, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO); pos = Z[f"{o}|pos"]
        for b in [b for q, b in av if q == o]:
            R1[j] = SC.pscore(Z[f"{o}|{b}|filt"]); assert np.allclose(R1[j], Z[f"{o}|{b}|sc_filt"]); pp = np.load(REPO / f"runs/causalpers/pp_{o}_b{b}.npz")
            for i, (fm, p, k, t0, R, cap) in enumerate(GRID):
                w = SC.weight(pos, t0, R, cap); lp = np.log(np.clip(np.nan_to_num(pp[f"{fm}|{p}|k{k}"], nan=1 / 3), 1e-6, 1))
                L = np.log(np.clip(Z[f"{o}|{b}|raw"], 1e-6, 1)) + w[:, None] * lp; L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
                G[i, j] = SC.pscore(K.gate(P, K.J.filt_joint(P, hc, "urn"), o, lat))
            j += 1
        print(o, "done", flush=True)
    D = SC.by_order(G - R1[None], av); Dc = np.stack([D[o] for o in orders], 1)
    print("\nfusion BEFORE the count filter, change against r15 + count filter")
    for i, g in enumerate(GRID):
        SC.summary(f"fixed {g}", {o: Dc[i, n] for n, o in enumerate(orders)})
    nP, picks = SC.nested(Dc, orders); SC.summary("NESTED filter(raw x personal)", nP); print("picks:", sorted({str(x) for x in picks}))


if __name__ == "__main__":
    main()
