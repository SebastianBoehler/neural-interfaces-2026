"""Tables for ledger s227 from runs/extharness/ret_<corpus>.pkl. Use 1: person-specific route strength (LORO on session 1)
against one common strength chosen leave-one-person-out; Use 2: gain by gap. 80 % intervals from 2000 person resamples.
Usage: python ret_report.py CORPUS[,CORPUS...] [TAG] [STAGE=tsem|member]"""
import os, sys, pickle
import numpy as np
from scipy.stats import spearmanr
OUT = os.path.abspath(__file__ + "/../../../../runs/extharness")


def boot(f, n, B=2000):
    r = np.random.default_rng(0); v = [f(r.integers(0, n, n)) for _ in range(B)]; v = np.array(v, float)
    return f(np.arange(n)), np.nanpercentile(v, 10), np.nanpercentile(v, 90)


def fm(t, sign=True):
    f = "{:+.2f}" if sign else "{:.2f}"; return (f + " [" + f + ", " + f + "]").format(*t)


def curves(res, gap, stage, scheme=None):
    k = {"member": 0, "tsem": 1}[stage]
    return np.array([np.mean([v["ba"][:, k] for (g, sc, _), v in r["test"].items() if g == gap and scheme in (None, sc)], 0)
                     for r in res if any(g == gap for g, _, _ in r["test"])])


def report(name, tag="", stage="tsem"):
    z = pickle.load(open(f"{OUT}/ret_{name}{tag}.pkl", "rb")); res, S = z["res"], np.array(z["s"]); n = len(res)
    T = curves(res, 1, stage); cv = np.array([r["cv"] for r in res]); adv = np.array([r["adv"] for r in res])
    scv, sbest = S[cv.argmax(1)], S[T.argmax(1)]
    L = [f"## {name}: {n} people, stage {stage}, gap 1 (mean over orders)"]
    L.append("s: " + " ".join(f"{s:g}" for s in S))
    L.append("gain vs s 0: " + " ".join(f"{d:+.2f}" for d in (T - T[:, :1]).mean(0)) + f" | level at s 0 {T[:, 0].mean():.1f}, "
             f"member s 0 {curves(res, 1, 'member')[:, 0].mean():.1f}")
    def common_lopo(ix):
        out = []
        for i in ix:
            m = np.ones(n, bool); m[i] = False; out.append(T[i, T[m].mean(0).argmax()])
        return np.array(out)
    lo = common_lopo(range(n)); pp = T[np.arange(n), cv.argmax(1)]; orc = T.max(1)
    L.append(f"policies (BA): none {T[:, 0].mean():.2f} | common (LOPO) {lo.mean():.2f} (in-sample best s "
             f"{S[T.mean(0).argmax()]:g}) | per-person s_cv {pp.mean():.2f} | oracle {orc.mean():.2f}")
    L.append(f"pp − common {fm(boot(lambda ix: (pp - lo)[ix].mean(), n))} ({((pp - lo) > 0).sum()} up, {((pp - lo) < 0).sum()} down) | "
             f"oracle − common {fm(boot(lambda ix: (orc - lo)[ix].mean(), n))}")
    sp = lambda a, b: (lambda ix: spearmanr(a[ix], b[ix])[0] if np.ptp(a[ix]) > 0 and np.ptp(b[ix]) > 0 else np.nan)  # noqa: E731
    g25 = T[:, list(S).index(0.25)] - T[:, 0]
    L.append(f"Spearman s_cv~best s {fm(boot(sp(scv, sbest), n))} | LORO advantage~best s {fm(boot(sp(adv, sbest), n))} | "
             f"advantage~gain at s .25 {fm(boot(sp(adv, g25), n))}")
    L.append(f"s_cv distribution {dict(zip(*np.unique(scv, return_counts=True)))} | best-s distribution {dict(zip(*np.unique(sbest, return_counts=True)))}")
    k0 = [v for r in res for (g, _, _), v in r["test"].items() if g == 1]
    L.append(f"identity: routed share {np.mean([v['routed'] for v in k0]):.2f}, right {np.mean([v['right'] for v in k0]):.2f}, "
             f"wrong {np.mean([v['wrong'] for v in k0]):.2f}; own head alone {np.mean([v['own_head'] for v in k0]):.1f}")
    for sc in ("rec", "full", "within"):
        Ts = curves(res, 1, stage, sc); L.append(f"  {sc}: gain at s .25 {(Ts[:, 4] - Ts[:, 0]).mean():+.2f}, best common s "
                                                f"{S[Ts.mean(0).argmax()]:g} ({Ts.mean(0).max() - Ts[:, 0].mean():+.2f})")
    gaps = sorted({g for r in res for g, _, _ in r["test"]})
    if len(gaps) > 1:
        sc_ = S.tolist().index(S[T.mean(0).argmax()])
        G = {g: curves(res, g, stage) for g in gaps}; m = len(G[gaps[-1]])
        for si, lab in ((sc_, f"common s {S[sc_]:g}"), (4, "s .25")):
            gains = {g: G[g][:m, si] - G[g][:m, 0] for g in gaps}
            L.append(f"Use 2, {lab}: " + " | ".join(f"gap {g} gain {fm(boot(lambda ix, g=g: gains[g][ix].mean(), m))}" for g in gaps)
                     + f" | gap {gaps[-1]} − gap 1 {fm(boot(lambda ix: (gains[gaps[-1]] - gains[1])[ix].mean(), m))}")
        L.append("Use 2, member at s 0 by gap: " + " ".join(f"{g}: {curves(res, g, 'member')[:, 0].mean():.1f}" for g in gaps)
                 + " | own head alone by gap: " + " ".join(f"{g}: {np.mean([v['own_head'] for r in res for (gg, _, _), v in r['test'].items() if gg == g]):.1f}" for g in gaps))
    return "\n".join(L)


if __name__ == "__main__":
    tag = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("STAGE") else ""
    st = next((a.split("=")[1] for a in sys.argv if a.startswith("STAGE=")), "tsem")
    print("\n\n".join(report(nm, tag, st) for nm in sys.argv[1].split(",")))
