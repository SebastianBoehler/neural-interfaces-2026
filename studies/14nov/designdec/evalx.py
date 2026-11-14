"""DESIGNDEC scoring (ledger s239). Per order, outputs in recording layout {name: (N, 3)}; 'off' is the unmodified chain.
Each output is scored alone, with the gated J-wprior/urn filter on top (s228c a: rundetect/package.py) and with the gated
true-run filter. Differences against off + the same filter, new people, per person (for bootstrap and collapse checks)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../rundetect"))
import common as C  # noqa: E402
import jointcount as J  # noqa: E402
import package as PK  # noqa: E402

FILTERS = ("none", "gJ", "gT")


def rank(order):
    sc = order.rstrip("0123456789"); g = C.PERMS[f"{sc}|{order[len(sc):]}|{C.p0}"]; rk = np.empty(C.N, int); rk[g] = np.arange(C.N); return rk


def filtered(order, P, cache):
    if "hc" not in cache:
        cache.update(hc=J.hyp_cache(order, "wprior", PK.X, PK.LAM), lat=PK.latch(order), rk=rank(order))
    FJ = J.filt_joint(P, cache["hc"], "urn"); FT = J.filt_groups(P, cache["rk"], C.RUNI, lambda r: J.TRUE_CAP[C.RUN[r[0]]])
    return {"none": P, "gJ": PK.gate(P, FJ, order, cache["lat"]), "gT": PK.gate(P, FT, order, cache["lat"])}


def person_scores(order, outs):
    """-> {(name, filter): (10,) per-person BA}"""
    cache, res = {}, {}
    for k, P in outs.items():
        for f, o in filtered(order, P.astype(np.float64), cache).items():
            res[k, f] = C.A.SC.person(o, C.p0)
    return res


def report(res, label=""):
    names = sorted({k for k, _ in res if k != "off"})
    print(f"  {label} off: " + " ".join(f"{f} {res['off', f].mean():.2f}" for f in FILTERS) + " (absolute)", flush=True)
    for k in names:
        cells = []
        for f in FILTERS:
            d = res[k, f] - res["off", f]; cells.append(f"{f} {d.mean():+.2f} (min {d.min():+.1f})")
        print(f"  {label} {k:10s} vs off+same filter: " + " | ".join(cells) + f" || vs off+gJ: {(res[k, 'gJ'] - res['off', 'gJ']).mean():+.2f}", flush=True)
