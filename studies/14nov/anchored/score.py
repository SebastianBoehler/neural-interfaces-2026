"""POPULATION-ANCHORED PERSONAL CLASSIC MODELS scorer (ledger s254). As score_pl.py: r15 chain (V6Fr, six bases), per
order and person the mean over sessions of the mean over the six pairs of the mean over bases of the cell-mean BA of the
chain alone or fused (product, equal weight) with a personal output; gain against the chain alone on the same order.
Systems per label form: members L (CS grid), D1 / D2 (KS grid), E = mean(L_i, D1_j, D2_j) on the joint (C, k) grid.
Nested pick: for person p the grid point with the best scheme-equal mean gain on the other nine people.
Usage: python studies/14nov/anchored/score.py [alpha] > evidence/anchored_14nov/score[_alpha].txt"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import anchor as A  # noqa: E402
import splits as S  # noqa: E402
from score_pl import ALL, SCH, boot, wmean  # noqa: E402

m = S.meta(); PAIRS = [(s, n, t) for s in S.sessions(m) for n, _, t in S.pairs(m, s)]
ALPHA = sys.argv[1:] == ["alpha"]  # amendment: scaled-anchor LR family La (alpha_*.npz) in place of L
LG = [f"a{a:g}" for a in A.AL] if ALPHA else [str(c) for c in A.CS]; LN = "La" if ALPHA else "L"
NAMES = [f"{LN} {c}" for c in LG] + [f"D1 k{k:g}" for k in A.KS] + [f"D2 k{k:g}" for k in A.KS] + [f"E{LN[1:]} {c}|k{k:g}" for c in LG for k in A.KS]


def ba(y, it, pred):
    """vectorised S.cell_score over leading dims of pred (..., n) -> (...) in %"""
    return 100 * np.mean([np.mean([(pred[..., (it == c) & (y == k)] == k).mean(-1) for k in np.unique(y[it == c])], 0) for c in np.unique(it)], 0)


def systems(z, form, s, n):
    """-> (len(NAMES), n_test, 3) personal outputs for one split"""
    L, D1, D2 = (z[f"{form}|{s}|{n}|{k}"].astype(np.float64) for k in (LN, "D1", "D2"))
    E = (L[:, None] + D1[None] + D2[None]) / 3; return np.concatenate([L, D1, D2, E.reshape(-1, *L.shape[1:])])


def scores(CH, Z, form):
    """-> chain (orders, 10), fused (orders, systems, 10), alone (orders, systems, 10)"""
    ch, fu, al = np.zeros((len(ALL), 10)), np.zeros((len(ALL), len(NAMES), 10)), np.zeros((len(ALL), len(NAMES), 10))
    for i, o in enumerate(ALL):
        z = Z["true" if form == "true" else o]; acc = {}
        for s, n, t in PAIRS:
            y, it = m["y"][t], m["interface"][t]; C = np.log(np.clip(np.stack([c[t] for c in CH[o]]), 1e-6, 1)); P = systems(z, form, s, n)
            f = ba(y, it, (C[None] + np.log(np.clip(P, 1e-6, 1))[:, None]).argmax(-1)).mean(1)
            acc.setdefault(s, []).append((ba(y, it, C.argmax(-1)).mean(), f, ba(y, it, P.argmax(-1))))
        for j, p in enumerate(S.PEOPLE):
            v = [acc[s] for s in acc if s.startswith(p)]
            ch[i, j] = np.mean([np.mean([a[0] for a in x]) for x in v])
            fu[i, :, j] = np.mean([np.mean([a[1] for a in x], 0) for x in v], 0); al[i, :, j] = np.mean([np.mean([a[2] for a in x], 0) for x in v], 0)
    return ch, fu, al


def nested(G):
    """G (orders, systems, 10) gains -> (orders, 10) at each person's pick from the other nine, picks"""
    w = np.stack([wmean(G[:, k], ALL) for k in range(G.shape[1])]); pick = [int(np.argmax(np.delete(w, p, 1).mean(1))) for p in range(10)]
    return np.stack([G[:, pick[p], p] for p in range(10)], 1), pick


def line(name, D, lev=None):
    g = wmean(D, ALL); lo, hi = boot(D, ALL); sch = {q: np.mean([D[i].mean() for i, o in enumerate(ALL) if SCH(o) == q]) for q in ("within", "full", "runs")}
    return (f"  {name:16s} {'' if lev is None else f'{lev:5.1f}':5s} {g.mean():+5.2f} [{lo:+.2f}, {hi:+.2f}] {int((g > 0).sum()):2d}/10 | "
            + " ".join(f"{q[0]} {v:+.2f}" for q, v in sch.items()) + " | " + " ".join(f"{x:+.1f}" for x in g))


def main():
    CH = {o: [np.load(S.REPO / f"runs/chainout/s242r15_{o}.npz")[k].astype(np.float64) for k in sorted(np.load(S.REPO / f"runs/chainout/s242r15_{o}.npz").files) if k.endswith("|V6Fr")] for o in ALL}
    assert all(len(v) == 6 for v in CH.values())
    o, (s, n, t) = "within0", PAIRS[5]; P = np.random.default_rng(0).dirichlet(np.ones(3), len(t))
    assert abs(ba(m["y"][t], m["interface"][t], S.fuse(CH[o][0][t], P).argmax(1)) - S.cell_score(m, t, S.fuse(CH[o][0][t], P))) < 1e-9
    Z = {k: dict(np.load(A.OUT / f"{k}.npz")) for k in ALL + ["true"]}; first = True
    if ALPHA:
        for k, z in Z.items():
            za = np.load(A.OUT / f"alpha_{k}.npz"); z.update(za)
            assert all(np.abs(za[q][0] - z[q[:-1]][-1]).max() < 1e-6 for q in za.files), k  # a 0 == L pers
    for form in ("true", "hard", "soft", "filt"):
        ch, fu, al = scores(CH, Z, form); G = fu - ch[:, None]
        if first:
            print(f"r15 chain on the test runs, ten-person mean {wmean(ch, ALL).mean():.2f} (score_pl: 79.7); per person " + " ".join(f"{x:.1f}" for x in wmean(ch, ALL))); first = False
        print(f"\n=== labels: {form}.  system  alone | c x system gain [80 % people x orders] people up | scheme means | per person P01..P10")
        for k, nm in enumerate(NAMES):
            if not nm.startswith("E") or nm.endswith(("|k0", "|k10", "|k100")):
                print(line(nm, G[:, k], wmean(al[:, k], ALL).mean()))
        nl = len(LG)
        for fam, ix in ((LN, range(nl)), ("D1", range(nl, nl + 8)), ("D2", range(nl + 8, nl + 16)), ("E" + LN[1:], range(nl + 16, len(NAMES)))):
            ix = list(ix); D, pick = nested(G[:, ix]); print(line(f"{fam} nested", D) + "  picks " + ",".join(NAMES[ix[p]].split(" ", 1)[1] for p in pick))
        k = int(np.argmax(wmean(G, ALL).mean(1))); print(line("best hindsight", G[:, k]) + f"  ({NAMES[k]}; not a valid pick)")


if __name__ == "__main__":
    main()
