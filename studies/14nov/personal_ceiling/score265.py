"""s265 scorer. V1: c x bE^w, families alone, leave-one-family-out (the 16 base files, filt/hard/soft as score_pl).
V2: delivery-prefix nets (runs/personal_ceiling_v2/ftd_*.npz) against the r15 chain on the same windows (position >= 120).
V3: 4 seeds per family (base files + runs/personal_ceiling_v3).
Usage: python score265.py v1|v2|v3 > evidence/personal_ceiling_14nov/s265.txt"""
import glob, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import splits as S  # noqa: E402
import score_pl as P  # noqa: E402

FAM = ("fb", "fbl", "fbraw", "sinc"); LEO = S.REPO / "runs/personal_ceiling_leo"


def need(var, n, pat):
    """dir from env var, failing loudly if it is unset, missing or holds fewer than n files matching pat"""
    d = os.environ.get(var)
    if not d or not Path(d).is_dir() or len(glob.glob(str(Path(d) / pat))) < n:
        sys.exit(f"FATAL: ${var}={d!r} unset, missing or holds fewer than {n} {pat} files")
    return Path(d)


def fam_probs(dirs, form, fams, o, s, n, seeds="*"):
    g = 0 if o in P.GROUPS[0] else 1; fs = [f for d in dirs for fam in fams for sd in ([seeds] if isinstance(seeds, str) else seeds) for f in sorted(glob.glob(str(d / f"ftpl_{fam}_s{sd}_g{g}.npz")))]
    assert fs and len(set(fs)) == len(fs), (fams, seeds)
    Z = [np.load(f) for f in fs]
    return np.mean([z[f"W|{s}|{n}|{o}|{form}"] for z in Z], 0), len(Z)


def report(name, D, base, orders):
    g = P.wmean(D, orders) - P.wmean(base, orders); lo, hi = P.boot(D - base, orders)
    print(f"  {name:28s} {P.wmean(D, orders).mean():6.2f} | {g.mean():+5.2f} [{lo:+.2f}, {hi:+.2f}] {int((g > 0).sum()):2d}/10 | P04 {g[3]:+.2f} P10 {g[9]:+.2f}", flush=True)


def v1(dirs=(LEO,), tag="V1"):
    CH, LAB, PA = P.load(); orders = P.SIX; base = P.person_scores(CH, orders, lambda o, s, n: []); form = "filt"
    print(f"{tag}: r15 chain on the six bE orders {P.wmean(base, orders).mean():.2f}; form {form}; level | gain [80 % people x orders] people up | P04 P10")
    def fused(fams, w):
        return P.person_scores(CH, orders, lambda o, s, n: [fam_probs(dirs, form, fams, o, s, n)[0] ** w / 1.0 if w != 1 else fam_probs(dirs, form, fams, o, s, n)[0]])
    for w in (0.5, 1.0, 1.5):
        report(f"c x bE^{w} (4 fam)", fused(FAM, w), base, orders)
    for f in FAM:
        report(f"c x bE {f} alone", fused((f,), 1.0), base, orders)
    for f in FAM:
        report(f"c x bE without {f}", fused(tuple(x for x in FAM if x != f), 1.0), base, orders)
    print("  n files per cell:", fam_probs(dirs, form, FAM, orders[0], *[(s, n) for s, n, t in P.PAIRS][0])[1])


def v2():
    CH, LAB, PA = P.load(); d2 = need("S265_V2_DIR", 8, "ftd_*_s*.npz"); Z = [np.load(f) for f in sorted(glob.glob(str(d2 / "ftd_*_s*.npz")))]; assert len(Z) == 8, len(Z); print("V2 dir:", d2)
    orders = P.SIX; ch = np.load(S.REPO / "runs/causalpers/chain.npz"); sess = S.sessions(P.m); res = {}
    for k in ("chain", "bE", "c x bE"):
        res[k] = np.zeros((len(orders), 10))
    for i, o in enumerate(orders):
        pos = ch[f"{o}|pos"]; per = {k: {} for k in res}
        for s in sess:
            idx = Z[0][f"D|{s}|{o}|idx"]; assert all((z[f"D|{s}|{o}|idx"] == idx).all() for z in Z) and (pos[idx] >= 120).all()
            pb = np.mean([z[f"D|{s}|{o}|p"] for z in Z], 0); sc = lambda Q: S.cell_score(P.m, idx, Q)  # noqa: E731
            r = {"chain": [], "bE": [sc(pb)], "c x bE": []}
            for c in CH[o]:
                r["chain"].append(sc(c[idx])); r["c x bE"].append(sc(S.fuse(c[idx], pb)))
            for k in res:
                per[k].setdefault(s[:3], []).append(np.mean(r[k]))
        for k in res:
            res[k][i] = [np.mean(per[k][p]) for p in S.PEOPLE]
    b = res["chain"]; print(f"V2 delivery prefix t=120 (windows from position 120 on): chain {P.wmean(b, orders).mean():.2f}; 6 orders x 8 nets")
    for k in ("bE", "c x bE"):
        report(k, res[k], b, orders)
    for sc_ in ("within", "full", "runs"):
        ix = [j for j, o in enumerate(orders) if o.startswith(sc_)]; g = (res["c x bE"][ix] - b[ix]).mean(); print(f"    scheme {sc_}: c x bE {g:+.2f}")


def s267():
    """s267: c x bE^0.5, fresh seeds 4-7 (S267_DIR, 20 files): S1 sinc x 4 seeds; S2 sinc + fb + fbl x seeds 4,5; S3 four families x seeds 4,5"""
    d = need("S267_DIR", 20, "ftpl_*_s[4-7]_g*.npz"); CH, LAB, PA = P.load(); orders = P.SIX; base = P.person_scores(CH, orders, lambda o, s, n: [])
    print(f"S267 dir {d}; chain {P.wmean(base, orders).mean():.2f}; filt, w 0.5, six orders; level | gain [80 % people x orders] people up | P04 P10")
    sets = {"S1 sinc x 4 seeds": (("sinc",), (4, 5, 6, 7)), "S2 sinc+fb+fbl x 2": (("sinc", "fb", "fbl"), (4, 5)), "S3 4 families x 2": (FAM, (4, 5))}
    out = {k: P.person_scores(CH, orders, lambda o, s, n, a=a: [fam_probs((d,), "filt", a[0], o, s, n, a[1])[0] ** 0.5]) for k, a in sets.items()}
    for k, D in out.items():
        report(k, D, base, orders)
    report("S2 - S3 (vs S3 as base)", out["S2 sinc+fb+fbl x 2"], out["S3 4 families x 2"], orders)


if __name__ == "__main__":
    {"v1": v1, "v2": v2, "v3": lambda: v1((LEO, need("S265_V3_DIR", 16, "ftpl_*_s[23]_g*.npz")), "V3 (4 seeds)"), "s267": s267}[sys.argv[1]]()
