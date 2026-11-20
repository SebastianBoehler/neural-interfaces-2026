"""Collect lane T3: official cell-mean BA per base and variant (mean of the streamed orders), people-resampled 80 % intervals,
paired differences, ms/window. Local, numpy only.

python cleanslate/ttt_base/collect.py WORKDIR --tags a0,a1,... [--variants A0,A1,A2,A2h] [--pairs a1:a0,b1:b0] [--json OUT]
WORKDIR holds TAG/eval_Pxx.npz (stream_eval output). A base needs all ten people. Cell value = mean over the orders of that cell's BA;
interval = people resampled (10k draws, cell-weighted mean), differences paired by cell. TTT gain = (variant - A1) per base.
Late fusion (member e): --fuse DIR --w 0.25,0.5,1 adds rows "<var>+M<w>" = argmax((1-w) p_base + w p_MANAS), MANAS posteriors from
DIR/M_Pxx.npz (manas_stream.py; same rows and orders). The TTT teacher never sees MANAS; w=1 is MANAS alone.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "ref"))
import score as SC  # noqa: E402  (vendored team C scorer: cells, boot)

PEOPLE = SC.PEOPLE


def per_person(d, Z, var, post_fn=None):
    """{person: (keys, values mean over orders)} for one variant."""
    out, secs = {}, []
    for p in PEOPLE:
        z = Z[p]
        orders = sorted({k.split("|")[1] for k in z.files if k.startswith(f"{var}|") and k.endswith("|label")})
        vals = None
        for o in orders:
            rows = z[f"rows|{o}"]
            lab = post_fn(p, o, z) if post_fn else z[f"{var}|{o}|label"]
            c = SC.cells(d, rows, lab)
            keys = list(c)
            v = np.array([c[k] for k in keys])
            vals = v[None] if vals is None else np.concatenate([vals, v[None]])
            secs.append(z[f"{var}|{o}|sec"]) if var != "M" else None
        out[p] = (keys, vals.mean(0))
    return out, (1e3 * float(np.concatenate(secs).mean()) if secs else float("nan"))


def interval(pp, n=10000, seed=0):
    return (100 * SC.boot([pp[p][1] for p in PEOPLE], n, seed)).round(2)


def diff(a, b, n=10000, seed=0):
    dp = []
    for p in PEOPLE:
        assert a[p][0] == b[p][0], "cell sets differ"
        dp.append(a[p][1] - b[p][1])
    allc = np.concatenate(dp)
    return 100 * allc.mean(), (100 * SC.boot(dp, n, seed)).round(2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--variants", default="A0,A1,A2,A2h")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--json")
    ap.add_argument("--vpairs", default="", help="TAG/VARIANT:TAG/VARIANT paired differences, e.g. a0/A2+M0.25:a0/A2")
    ap.add_argument("--fuse", default="")
    ap.add_argument("--w", default="0.25,0.5")
    a = ap.parse_args()
    M = {p: np.load(Path(a.fuse) / f"M_{p}.npz") for p in PEOPLE} if a.fuse else None
    d = np.load(SC.ROOT / "data/proteus_windows.npz")
    res, store = {}, {}
    for tag in a.tags.split(","):
        Z = {p: np.load(Path(a.work) / tag / f"eval_{p}.npz") for p in PEOPLE}
        row = {}
        for v in a.variants.split(","):
            if not any(k.startswith(v + "|") for k in Z[PEOPLE[0]].files):
                continue  # variant not run for this base
            pp, ms = per_person(d, Z, v)
            allc = np.concatenate([pp[p][1] for p in PEOPLE])
            store[(tag, v)] = pp
            row[v] = dict(ba=round(100 * float(allc.mean()), 2), ci=interval(pp).tolist(), ms=round(ms, 1), n_cells=len(allc))
        for v in ([v for v in a.variants.split(",") if v in row] if M else []):
            for w in map(float, a.w.split(",")):
                def fl(p, o, z, v=v, w=w):
                    assert np.array_equal(M[p][f"rows|{o}"], z[f"rows|{o}"]), "row order differs"
                    return ((1 - w) * z[f"{v}|{o}|post"] + w * M[p][f"M|{o}|post"]).argmax(1)
                pp, _ = per_person(d, Z, v, fl)
                allc = np.concatenate([pp[p][1] for p in PEOPLE])
                store[(tag, f"{v}+M{w}")] = pp
                row[f"{v}+M{w}"] = dict(ba=round(100 * float(allc.mean()), 2), ci=interval(pp).tolist(), ms=float("nan"), n_cells=len(allc))
        a1 = store[(tag, "A1")] if "A1" in a.variants else None
        for v in [v for v in a.variants.split(",") if v in row]:
            if a1 is not None and v not in ("A0", "A1"):
                g, ci = diff(store[(tag, v)], a1)
                row[v]["gain_vs_A1"], row[v]["gain_ci"] = round(g, 2), ci
        res[tag] = row
        print(tag, " | ".join(f"{v} {r['ba']:.2f} {r['ci']} {r['ms']:.0f}ms" + (f" gain {r['gain_vs_A1']:+.2f} {r['gain_ci']}" if "gain_vs_A1" in r else "") for v, r in row.items()))
    for pr in filter(None, a.pairs.split(",")):
        x, y = pr.split(":")
        for v in a.variants.split(","):
            g, ci = diff(store[(x, v)], store[(y, v)])
            res.setdefault("pairs", {})[f"{x}-{y}|{v}"] = dict(diff=round(g, 2), ci=ci)
            print(f"{x} - {y} [{v}] {g:+.2f} {ci}")
    for pr in filter(None, a.vpairs.split(",")):
        x, y = pr.split(":")
        g, ci = diff(store[tuple(x.split("/"))], store[tuple(y.split("/"))])
        res.setdefault("vpairs", {})[pr] = dict(diff=round(g, 2), ci=ci)
        print(f"{x} - {y} {g:+.2f} {ci}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
