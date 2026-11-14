"""Table for ledger s216 from runs/extharness/res_<corpus>.npz: per scheme (seeds averaged per person first), person-equal
balanced accuracy per stage with 80 % person-resampled intervals, stage deltas, and the rebalancing-window check.
Usage: python report.py CORPUS[,CORPUS...] [TAG]"""
import os, sys
import numpy as np
OUT = os.path.abspath(__file__ + "/../../../../runs/extharness")
STAGES = ("member", "rebal", "sessionem", "tsem"); SCHEMES = ("rec", "full", "runs", "within")


def boot(x, B=2000):
    x = np.asarray(x); r = np.random.default_rng(0); m = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 10), np.percentile(m, 90)


def fmt(x, sign=False):
    m, lo, hi = boot(x); f = "{:+.2f}" if sign else "{:.1f}"
    return (f + " [" + f + ", " + f + "]").format(m, lo, hi)


def table(name, tag=""):
    z = np.load(f"{OUT}/res_{name}{tag}.npz"); keys = [k for k in z.files if k != "people"]
    get = lambda sc, W, st: np.mean([z[k] for k in keys if k.split("|")[0] == sc and k.split("|")[2] == str(W)
                                      and k.split("|")[3] == st], 0)
    n = len(z["people"]); Ws = sorted({int(k.split("|")[2]) for k in keys})
    lines = [f"## {name}: {n} people, person-equal BA, 80 % person intervals; W {Ws}"]
    lines.append("| scheme | member | rebal | SessionEM | TSEM | rebal−member | EM−rebal | TSEM−EM | chain−member |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for sc in SCHEMES:
        v = {st: get(sc, 30, st) for st in STAGES}
        lines.append(f"| {sc} | " + " | ".join(fmt(v[s]) for s in STAGES) + " | "
                     + " | ".join(fmt(v[b] - v[a], True) for a, b in (("member", "rebal"), ("rebal", "sessionem"),
                                                                     ("sessionem", "tsem"), ("member", "tsem"))) + " |")
    lines.append("")
    lines.append("| scheme | " + " | ".join(f"chain W{w}−W30 (up/n)" for w in Ws if w != 30)
                 + " | " + " | ".join(f"rebal W{w}−W30" for w in Ws if w != 30) + " |")
    lines.append("|---|" + "---|" * (2 * (len(Ws) - 1)))
    for sc in SCHEMES:
        t30, b30 = get(sc, 30, "tsem"), get(sc, 30, "rebal"); row = []
        for w in Ws:
            if w != 30:
                d = get(sc, w, "tsem") - t30; row.append(f"{fmt(d, True)} ({(d > 0).sum()}/{n})")
        row += [fmt(get(sc, w, "rebal") - b30, True) for w in Ws if w != 30]
        lines.append(f"| {sc} | " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    print("\n\n".join(table(nm, tag) for nm in sys.argv[1].split(",")))
