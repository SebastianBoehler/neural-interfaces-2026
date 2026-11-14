"""s260 scorer: the ONE fixed form (cq50, a2, t0 90, R 120, cap 0.10, track F) on a fresh r15 draw, k 20 and k 10.
DRAW=d1|d2 python studies/14nov/causalpers/score260.py     -> prints the draw, saves runs/causalpers/s260_<draw>.npz
python studies/14nov/causalpers/score260.py pool           -> prints both draws pooled (needs both npz)
Zero-change: cap 0 equals r15 + filter (replay person scores in chain<draw>.npz) on every order x base, else fails."""
import os, sys
from pathlib import Path
os.environ.setdefault("SEL", "only"); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import score as SC  # noqa: E402

NAME = "cq50|a2"; T0, R, CAP = 90, 120, 0.10; REPO = SC.REPO


def draw():
    D = SC.DRAW; assert D in ("d1", "d2"), "set DRAW=d1 or d2 (draw 0 is §253's, never rescored here)"
    ORD, res = SC.ORD, {k: [] for k in (10, 20)}; zc = 0
    for o in ORD:
        d = {k: [] for k in (10, 20)}
        for b in range(6):
            f = REPO / f"runs/causalpers/pp_{o}_b{b}_cq50_pa2_{D}.npz"; assert f.exists(), f
            pz = np.load(f); assert str(pz["draw"]) == D, ("pp file draw", f, str(pz["draw"]))
            pos = SC.Z[f"{o}|pos"]; C = SC.Z[f"{o}|{b}|filt"]; ref = SC.pscore(C); assert np.allclose(ref, SC.Z[f"{o}|{b}|sc_filt"], atol=1e-9), ("zero-change", o, b)
            LC = np.log(np.clip(C, 1e-6, 1)); w0 = SC.weight(pos, T0, R, 0.0); assert np.allclose(SC.pscore(LC + w0[:, None] * 0), ref); zc += 1
            for k in (10, 20):
                v = pz[f"{NAME}|k{k}"]; nan = np.isnan(v[:, 0]); lp = np.log(np.clip(np.nan_to_num(v, nan=1 / 3), 1e-6, 1)); w = SC.weight(pos, T0, R, CAP)
                assert not (w[nan] > 0).any(); assert (np.isnan(pz[f"{NAME}|k{k}"][pos < 20, 0])).all()
                d[k].append(SC.pscore(LC + w[:, None] * lp) - ref)
        for k in (10, 20):
            res[k].append(np.mean(d[k], 0))
    print(f"s260 draw {D}: zero-change OK on {zc} order x base (scorer equals replay r15 + filter; cap 0 identical)")
    np.savez(REPO / f"runs/causalpers/s260_{D}.npz", k10=np.array(res[10]), k20=np.array(res[20]), orders=np.array(ORD))
    return {k: dict(zip(ORD, np.array(res[k]))) for k in (10, 20)}


def show(title, R2):
    print(f"\n{title}\n  change vs r15 + filter: scheme-equal | per scheme (orders) | [80 % people x orders] | people up | P01..P10")
    for k in (20, 10):
        SC.summary(f"k {k}" + (" (primary)" if k == 20 else " (reported)"), R2[k])


if __name__ == "__main__":
    if sys.argv[1:] == ["pool"]:
        zs = {d: np.load(REPO / f"runs/causalpers/s260_{d}.npz") for d in ("d1", "d2")}; O = list(zs["d1"]["orders"])
        for d in zs:
            show(f"draw {d}", {k: dict(zip(O, zs[d][f"k{k}"])) for k in (10, 20)})
        show("pooled (draws averaged per order x person; interval resamples people and orders)", {k: {o: (zs["d1"][f"k{k}"][i] + zs["d2"][f"k{k}"][i]) / 2 for i, o in enumerate(O)} for k in (10, 20)})
        print("\ndraw difference (d1 - d2), scheme-equal:", " ".join(f"k{k} {np.mean([ (zs['d1'][f'k{k}'][[j for j,o in enumerate(O) if o.rstrip('0123456789')==s]] - zs['d2'][f'k{k}'][[j for j,o in enumerate(O) if o.rstrip('0123456789')==s]]).mean() for s in SC.SCHEMES]):+.2f}" for k in (10, 20)))
    else:
        show(f"draw {SC.DRAW}", draw())
