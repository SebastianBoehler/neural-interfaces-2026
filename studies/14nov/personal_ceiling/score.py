"""PERSONAL CEILING scorer (ledger s246). Personal systems (classic.npz: a1-a4, aE; ft_*.npz: bE = mean of every
available family x seed file) against the population chain on exactly the same test windows: s242ctl V6Fr (new-person
path, 20 orders x 6 bases, recording layout) and, for P04 / P10 sessions 2-3, ret ER (routed returning path, 16 orders x
6 bases, seen_session layout). Fusion = product of experts per order and base, then the score is averaged.
Per person: mean over sessions of the mean over the six pairs. Intervals: 80 % people bootstrap of person-level changes.
Usage: python studies/14nov/personal_ceiling/score.py [FT=ft200] > evidence/personal_ceiling_14nov/score.txt"""
import glob, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import splits as S  # noqa: E402

FT = os.environ.get("FT", "ft200"); R = S.REPO; m = S.meta(); N = len(m["y"])
ORD = [f"within{i}" for i in range(12)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]
RORD = [f"within{i}" for i in range(8)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]
SCH = lambda o: o.rstrip("0123456789")  # noqa: E731
WTS = {len(L) * 6: np.repeat([1 / sum(SCH(q) == SCH(o) for q in L) for o in L], 6) for L in (ORD, RORD)}  # schemes weighted equally (6 bases per order)


def chain():
    out = []
    for o in ORD:
        z = np.load(R / f"runs/chainout/s242ctl_{o}.npz"); out += [z[k].astype(np.float64) for k in sorted(z.files) if k.endswith("|V6Fr")]
    assert len(out) == 120 and out[0].shape == (N, 3), len(out); return np.stack(out)


def ret():
    idx = np.load(R / "runs/replay_data/bases.npz", allow_pickle=True)["idx|seen_session"].astype(int); out = []
    for o in RORD:
        z = np.load(R / f"runs/chainout/ret_{o}.npz")
        for k in sorted(k for k in z.files if k.endswith("|ER")):
            P = np.full((N, 3), np.nan); P[idx] = z[k]; out.append(P)
    assert len(out) == 96, len(out); return np.stack(out)


def personal():
    z = dict(np.load(S.OUT / "classic.npz")); P = {k: v for k, v in z.items()}
    fs = sorted(glob.glob(str(S.OUT / "ft_*_s*.npz"))); fams = sorted({os.path.basename(f).split("_")[1] for f in fs})
    if fs:
        F = [dict(np.load(f)) for f in fs]; keys = [k for k in F[0] if k.endswith("|idx")]
        for k in keys:
            b = k[:-4]; assert all((f[k] == F[0][k]).all() for f in F), k
            for v in ("base", "ft60", "ft200", "ft400"):
                P[f"{b}|b_{v}"] = np.mean([f[f"{b}|{v}"] for f in F], 0)
            P[f"{b}|bE"] = P[f"{b}|b_{FT}"]
    print(f"nets: {len(fs)} files, families {fams}" if fs else "nets: none yet")
    return P


def sc_many(CH, rows, Q=None):
    """mean score over the chain's orders x bases, optionally fused with personal Q"""
    return float(np.average([S.cell_score(m, rows, S.fuse(c[rows], Q) if Q is not None else c[rows]) for c in CH], weights=WTS[len(CH)]))


def boot(d, rng=np.random.default_rng(0)):
    b = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(4000)]; return np.percentile(b, [10, 90])


def main():
    P = personal(); CH = chain(); RT = ret(); hasb = any(k.endswith("|bE") for k in P)
    sysP = ["a1", "a2", "a3", "a4", "aE"] + (["b_base", "b_ft60", "b_ft200", "b_ft400"] if hasb else [])
    fus = ["c*aE"] + (["c*bE", "c*aE*bE"] if hasb else [])
    cols = ["chain"] + sysP + fus; rowsW = {}  # (session, pair) -> {col: score}
    for s in S.sessions(m):
        for name, _, tst in S.pairs(m, s):
            k = f"W|{s}|{name}"; assert (P[k + "|idx"] == tst).all(); r = {"chain": sc_many(CH, tst)}
            for v in sysP:
                r[v] = S.cell_score(m, tst, P[f"{k}|{v}"])
            r["c*aE"] = sc_many(CH, tst, P[k + "|aE"])
            if hasb:
                r["c*bE"] = sc_many(CH, tst, P[k + "|bE"]); r["c*aE*bE"] = sc_many(CH, tst, S.fuse(P[k + "|aE"], P[k + "|bE"]))
            if s[-1] != "1":
                r["ER"] = sc_many(RT, tst); r["ER*aE"] = sc_many(RT, tst, P[k + "|aE"])
                if hasb:
                    r["ER*bE"] = sc_many(RT, tst, P[k + "|bE"])
            rowsW[s, name] = r
    per = lambda p, c, prs=None: np.mean([np.mean([v[c] for (s, n), v in rowsW.items() if s == ss and (prs is None or n in prs)]) for ss in S.sessions(m) if ss.startswith(p)])  # noqa: E731
    print(f"\nWITHIN SESSION (train 4 runs, test 2 online runs; mean over 6 pairs, then sessions), cell-mean BA %, chain = s242ctl V6Fr")
    print("person " + " ".join(f"{c:>8s}" for c in cols))
    T = {c: np.array([per(p, c) for p in S.PEOPLE]) for c in cols}
    for i, p in enumerate(S.PEOPLE):
        print(f"{p:6s} " + " ".join(f"{T[c][i]:8.1f}" for c in cols))
    print(f"{'mean':6s} " + " ".join(f"{T[c].mean():8.1f}" for c in cols))
    for prs, lab in ((("34",), "test = last two online runs"), (("12",), "test = online runs 1+2")):
        print(f"{lab:30s} " + " ".join(f"{c} {np.mean([per(p, c, prs) for p in S.PEOPLE]):.1f}" for c in cols))
    print("\nchanges against the chain (ten people, 80 % people bootstrap) | people up")
    for c in cols[1:]:
        d = T[c] - T["chain"]; lo, hi = boot(d); print(f"  {c:9s} {d.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}] {int((d > 0).sum())}/10")
    print("\nPER SESSION (mean over the six pairs)")
    sc = ["chain", "ER", "aE"] + (["bE"] if hasb else []) + ["c*aE"] + (["c*bE"] if hasb else []) + ["ER*aE"] + (["ER*bE"] if hasb else [])
    print("session " + " ".join(f"{c:>7s}" for c in sc))
    for s in S.sessions(m):
        vs = [v for (ss, n), v in rowsW.items() if ss == s]
        print(f"{s:7s} " + " ".join(f"{np.mean([v[c if c != 'bE' else 'b_' + FT] for v in vs]):7.1f}" if all((c if c != 'bE' else 'b_' + FT) in v for v in vs) else f"{'-':>7s}" for c in sc))
    print("\nCROSS SESSION (P04 / P10: same personal system trained on all of session 1) vs WITHIN, same test windows")
    xs = ["aE"] + (["b_base", "bE"] if hasb else [])
    for p, s1, s in S.cross(m):
        line = []
        for v in xs:
            w = np.mean([S.cell_score(m, tst, P[f"W|{s}|{n}|{v}"]) for n, _, tst in S.pairs(m, s)])
            x = np.mean([S.cell_score(m, tst, P[f"X|{s}|{n}|{v}"]) for n, _, tst in S.pairs(m, s)])
            line.append(f"{v} W {w:.1f} X {x:.1f} drop {x - w:+.1f}")
        a = P[f"X|{s}|all|idx"]; whole = " ".join(f"{v} {S.cell_score(m, a, P[f'X|{s}|all|{v}']):.1f}" for v in xs)
        er = np.mean([sc_many(RT, tst) for _, _, tst in S.pairs(m, s)])
        print(f"  {s1}->{s}: " + " | ".join(line) + f" | ER {er:.1f} | X whole session: {whole}, ER {sc_many(RT, a):.1f}")


if __name__ == "__main__":
    main()
