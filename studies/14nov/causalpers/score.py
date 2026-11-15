"""CAUSAL PERSONALISATION scorer (ledger s253). Per order and base: out = chain x personal^w(t) (argmax of the log sum; the
renormalisation does not change it), w = 0 before t0, then cap x min(1, (t - t0 + 1) / R) (R 0: step), t = delivery position
in the session. Track R fuses into the raw r15 chain, track F into its count-filtered output. Person scores are the replay
cell-mean BA (checked equal to replay score.person on every order and base: zero-change). Changes are per person, mean
over bases, against r15 raw ('raw') and r15 + count filter ('filt'). Nested: for each person the grid point with the best
scheme-equal mean change on the other nine people. 80 % intervals resample people and orders (stack/score.py).
Usage: [EXT=1] python studies/14nov/causalpers/score.py > evidence/causalpers_14nov/score[_ext].txt
SEL=only scores the selection forms of amendment 3 (pp_*_<SELF>.npz) instead of the plain ones; SEL=all both. EXT=1 adds caps 0.1 / 0.15 and R 120 to the grid (declared after the first files; secondary)."""
import itertools, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from run import KS  # noqa: E402

ORD = [f"within{i}" for i in range(12)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]; SCHEMES = ("within", "full", "runs")
PIPES = ("l1", "a2", "lE"); T0S = (20, 40, 60, 90); RS = (0, 40, 80); CAPS = (0.25, 0.5, 0.75, 1.0); EXT = os.environ.get("EXT") == "1"
SEL = os.environ.get("SEL", ""); SELFORMS = ("top", "cq50", "cq25", "agree", "agcq"); SELF = "_" + "_".join(SELFORMS)
assert SEL in ("", "only", "all"), SEL
FORMS = (() if SEL == "only" else ("hard", "soft", "filt")) + (SELFORMS if SEL else ()); SFX = ("", SELF) if SEL == "all" else ((SELF,) if SEL else ("",))
fsrc = lambda fm: SELF if fm in SELFORMS else ""  # noqa: E731
if EXT:  # declared extension after the first two files showed picks at the grid edge (cap 0.25, R 80); secondary only
    RS, CAPS = RS + (120,), (0.1, 0.15) + CAPS
GRID = list(itertools.product(FORMS, PIPES, KS, T0S, RS, CAPS)); FIXED = ("filt", "lE", 10, 40, 40, 0.5) if "filt" in FORMS else ("cq50", "lE", 10, 40, 40, 0.5 if not EXT else 0.15)
DRAW = os.environ.get("DRAW", ""); assert DRAW in ("", "d1", "d2"), DRAW  # s260: fresh draws read only their own chain and pp files
Z = np.load(REPO / f"runs/causalpers/chain{DRAW}.npz"); assert str(Z["draw"]) == DRAW, ("chain file draw", str(Z["draw"]), DRAW)
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]
Y = Z["y"].astype(int); CELL = np.unique(Z["cell"], return_inverse=True)[1]; NC = CELL.max() + 1
CP = np.array([PEOPLE.index(Z["person"][CELL == c][0]) for c in range(NC)]); PW = np.stack([(CP == p) / (CP == p).sum() for p in range(10)])  # (10, NC)
TOT = np.bincount(CELL * 3 + Y, minlength=3 * NC).reshape(NC, 3); PRES = TOT > 0


def pscore(L):
    """L (N, 3) scores -> (10,) person cell-mean BA, as replay score.person"""
    hit = np.bincount(CELL * 3 + Y, weights=(L.argmax(1) == Y), minlength=3 * NC).reshape(NC, 3)
    ba = 100 * np.where(PRES, hit / np.maximum(TOT, 1), 0).sum(1) / PRES.sum(1); return PW @ ba


def weight(pos, t0, R, cap):
    return cap * np.clip((pos - t0 + 1) / R, 0, 1) if R else cap * (pos >= t0)


def available():
    return [(o, b) for o in ORD for b in range(6) if all((REPO / f"runs/causalpers/pp_{o}_b{b}{x}.npz").exists() for x in SFX)]


def compute(av):
    """-> ref {(o, b, 'raw'|'filt'): (10,)}, G {(track): (len(GRID), len(av), 10)}"""
    ref, G = {}, {t: np.zeros((len(GRID), len(av), 10)) for t in ("R", "F")}
    for j, (o, b) in enumerate(av):
        pos = Z[f"{o}|pos"]; LP = {}; pz = {x: np.load(REPO / f"runs/causalpers/pp_{o}_b{b}{x}.npz") for x in SFX}
        for tr, src in (("R", "raw"), ("F", "filt")):
            C = Z[f"{o}|{b}|{src}"]; ref[o, b, src] = s = pscore(C)
            assert np.allclose(s, Z[f"{o}|{b}|sc_{src}"], atol=1e-9), ("zero-change failed", o, b, src)
            LC = np.log(np.clip(C, 1e-6, 1))
            for i, (fm, p, k, t0, R, cap) in enumerate(GRID):
                if (fm, p, k) not in LP:
                    v = pz[fsrc(fm)][f"{fm}|{p}|k{k}"]; LP[fm, p, k] = (np.log(np.clip(np.nan_to_num(v, nan=1 / 3), 1e-6, 1)), np.isnan(v[:, 0]))
                lp, nan = LP[fm, p, k]; w = weight(pos, t0, R, cap); assert not (w[nan] > 0).any(), "weight on a window without a prediction"
                G[tr][i, j] = pscore(LC + w[:, None] * lp)
    return ref, G


def by_order(X, av):
    """(..., len(av), 10) -> {order: (..., 10)} mean over the bases present"""
    os_ = list(dict.fromkeys(o for o, _ in av)); return {o: X[..., [j for j, (q, _) in enumerate(av) if q == o], :].mean(-2) for o in os_}


def summary(name, D, rng=np.random.default_rng(0)):
    """D {order: (10,)} change -> printed line; scheme-equal mean, per scheme, interval over people x orders, people up"""
    by = {s: np.stack([v for o, v in D.items() if o.rstrip("0123456789") == s]) for s in SCHEMES if any(o.rstrip("0123456789") == s for o in D)}
    pt = np.mean([d.mean() for d in by.values()]); pm = np.mean([d.mean(0) for d in by.values()], 0); bt = []
    for _ in range(4000):
        pp = rng.integers(0, 10, 10); bt.append(np.mean([d[rng.integers(0, len(d), len(d))][:, pp].mean() for d in by.values()]))
    lo, hi = np.percentile(bt, [10, 90])
    print(f"  {name:34s} {pt:+.2f} | " + " ".join(f"{s} {d.mean():+.2f} ({len(d)})" for s, d in by.items()) + f" | [{lo:+.2f}, {hi:+.2f}] | {int((pm > 0).sum())}/10 | " + " ".join(f"{x:+.1f}" for x in pm))
    return pt


def sch_mean(X, orders):
    """X (cfg, orders, 10) -> (cfg, 10) scheme-equal mean over orders"""
    w = np.array([1 / sum(q.rstrip("0123456789") == o.rstrip("0123456789") for q in orders) for o in orders]); return np.tensordot(w / w.sum(), X, axes=(0, 1))


def nested(Dc, orders, mask=None):
    """Dc (cfg, orders, 10) change -> ({order: (10,)} held-out person's change at the grid point chosen on the other nine, picks)"""
    M = sch_mean(Dc, orders); idx = np.arange(len(GRID)) if mask is None else np.flatnonzero(mask); out = np.zeros(Dc.shape[1:]); picks = []
    for p in range(10):
        oth = [q for q in range(10) if q != p]; c = idx[np.argmax(M[idx][:, oth].mean(1))]; out[:, p] = Dc[c, :, p]; picks.append(GRID[c])
    return {o: out[i] for i, o in enumerate(orders)}, picks


def diag(av):
    """redundancy with the chain: on windows at delivery position >= 40, accuracy of the personal model alone and of the
    chain (raw / filt) on the same windows, and their argmax agreement (mean over order x base)"""
    print("\nredundancy (windows at position >= 40): personal alone acc | chain acc | agreement")
    for fm in FORMS:
        src = "filt" if fm == "filt" else "raw"
        for p in PIPES:
            v = []
            for o, b in av:
                m = Z[f"{o}|pos"] >= 40; P = np.load(REPO / f"runs/causalpers/pp_{o}_b{b}{fsrc(fm)}.npz")[f"{fm}|{p}|k10"][m].argmax(1); C = Z[f"{o}|{b}|{src}"][m].argmax(1)
                v.append((np.mean(P == Y[m]), np.mean(C == Y[m]), np.mean(P == C)))
            print(f"  {fm:4s} {p:3s} " + " | ".join(f"{x:.3f}" for x in np.mean(v, 0)))


def main():
    av = available(); orders = list(dict.fromkeys(o for o, _ in av))
    print(f"s253 CAUSAL PERSONALISATION: {len(av)} order x base files, {len(orders)} orders; zero-change check of the scorer on every one")
    ref, G = compute(av); print("ZERO-CHANGE OK: scorer equals replay score.person on r15 raw and filt")
    R0 = by_order(np.stack([ref[o, b, "raw"] for o, b in av])[None], av); R1 = by_order(np.stack([ref[o, b, "filt"] for o, b in av])[None], av)
    GR, GF = by_order(G["R"], av), by_order(G["F"], av)
    DR = np.stack([GR[o] - R0[o] for o in orders], 1); DF = np.stack([GF[o] - R1[o] for o in orders], 1); DF0 = np.stack([GF[o] - R0[o] for o in orders], 1)
    print("levels (scheme-equal): r15 raw %.2f, r15 + filter %.2f" % (sch_mean(np.stack([R0[o] for o in orders], 1), orders).mean(), sch_mean(np.stack([R1[o] for o in orders], 1), orders).mean()))
    print("\nchange: scheme-equal mean | per scheme (orders) | [80 % people x orders] | people up | per person P01..P10")
    summary("r15 + filter vs r15 raw", {o: R1[o][0] - R0[o][0] for o in orders})
    fi = GRID.index(FIXED)
    summary("FIXED F vs r15+filter", {o: DF[fi, i] for i, o in enumerate(orders)}); summary("FIXED F vs r15 raw", {o: DF0[fi, i] for i, o in enumerate(orders)})
    summary("FIXED R vs r15 raw", {o: DR[fi, i] for i, o in enumerate(orders)})
    nF, pF = nested(DF, orders); summary("NESTED F vs r15+filter (primary)", nF)
    summary("NESTED F vs r15 raw (same picks)", {o: v + (R1[o][0] - R0[o][0]) for o, v in nF.items()})
    nR, pR = nested(DR, orders); summary("NESTED R vs r15 raw", nR)
    for p in PIPES:
        summary(f"nested F, pipeline {p} only", nested(DF, orders, np.array([g[1] == p for g in GRID]))[0])
    for fm in FORMS:
        summary(f"nested F, labels {fm} only", nested(DF, orders, np.array([g[0] == fm for g in GRID]))[0])
    for k in KS:
        summary(f"nested F, l1 k {k} only", nested(DF, orders, np.array([g[1] == "l1" and g[2] == k for g in GRID]))[0])
    MF = sch_mean(DF, orders).mean(1); MR = sch_mean(DR, orders).mean(1); bF, bR = int(np.argmax(MF)), int(np.argmax(MR))
    summary(f"HINDSIGHT F {GRID[bF]}", {o: DF[bF, i] for i, o in enumerate(orders)}); summary(f"HINDSIGHT R {GRID[bR]}", {o: DR[bR, i] for i, o in enumerate(orders)})
    print("\nnested picks, track F (held-out person -> form, pipe, k, t0, R, cap):"); [print(f"  {PEOPLE[p]} {pF[p]}") for p in range(10)]
    print("nested picks, track R:", sorted({str(x) for x in pR}))
    print("\ntrack F grid marginals (best scheme-equal mean over the rest of the grid, all ten people; hindsight):")
    for j, (nm, vals) in enumerate((("form", FORMS), ("pipe", PIPES), ("k", KS), ("t0", T0S), ("R", RS), ("cap", CAPS))):
        print(f"  {nm:5s} " + " | ".join(f"{v}: {MF[[i for i, g in enumerate(GRID) if g[j] == v]].max():+.2f}" for v in vals))
    diag(av)
    np.savez(REPO / f"runs/causalpers/score{'_ext' if EXT else ''}{'_sel' + SEL if SEL else ''}.npz", DF=DF, DR=DR, DF0=DF0, orders=np.array(orders), grid=np.array([str(g) for g in GRID]))


if __name__ == "__main__":
    main()
