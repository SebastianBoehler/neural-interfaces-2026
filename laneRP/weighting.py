"""Unverified 60/6/40 reconstruction. Pair shared public identities and order draws."""
import numpy as np


def projections(groups, gate):
    arms = (("wave", "return_sch"), ("protected", "single_sch")) if gate == "wave" else (
        ("strict", "return_sch"), ("ridge", "return_sch"), ("neural", "return_sch"),
        ("loose", "return_sch"), ("hybrid", "single_sch"))
    pub = sorted({r["person"] for r in groups["new_pub"]})
    ret = sorted({r["person"] for r in groups["return_pub"]})
    single = sorted(set(pub) - set(ret))
    rng = np.random.default_rng(20261120)
    m = 20000

    def design(n, norders):
        return (rng.integers(n, size=(m, n, 1)),
                rng.integers(norders, size=(m, n, norders)))

    i8, o8 = design(8, 4)
    i2, o2 = design(2, 4)
    i9, o9 = design(9, 4)

    def values(rows, names, arm):
        vals = {}
        for r in rows:
            diff = r["ba"][arm] - r["ba"]["stock"]
            vals.setdefault((r["person"], r["order"]), []).extend(diff)
        ords = sorted({o for _, o in vals})
        return np.array([[np.mean(vals[p, o]) for o in ords] for p in names])

    def ci(x, b):
        return np.round([x, *np.quantile(b, [.1, .9])], 4).tolist()

    result = {}
    for arm, proxy in arms:
        nr = groups["new_pub"]
        a, b = values(nr, single, arm), values(nr, ret, arm)
        c = values(groups["return_pub"], ret, arm)
        ext = groups[proxy]
        ep = sorted({r["person"] for r in ext})
        # Keep full and within balanced, resampling each from its own four orders.
        extdraw, extpoint = 0., 0.
        for scheme in ("full", "within"):
            d = values([r for r in ext if r["order"][0] == scheme], ep, arm)
            eo = o9 if scheme == "full" else rng.integers(4, size=o9.shape)
            extdraw += .5 * d[i9, eo].mean((1, 2))
            extpoint += .5 * d.mean()
        # 8 single-visit and 2 three-visit held-new people: exact public official cell weighting.
        new = (8 * a.mean() + 6 * b.mean()) / 14
        newdraw = (8 * a[i8, o8].mean((1, 2)) + 6 * b[i2, o2].mean((1, 2))) / 14
        known = c.mean()
        knowndraw = c[i2, o2].mean((1, 2))
        intercept = (60 * new + 6 * known) / 106
        sample = (60 * newdraw + 6 * knowndraw) / 106
        result[arm] = dict(unknown_zero=ci(intercept, sample),
                           scherer_proxy=ci(intercept + 40/106*extpoint, sample + 40/106*extdraw),
                           returning_proxy=ci((6*known + 40*extpoint)/46, (6*knowndraw + 40*extdraw)/46),
                           eight_required_for_return_bar=round((46*2.5-6*known)/40, 4),
                           J_legacy=round(.57*new+.43*(.2*c[0].mean()+.8*c[1].mean()), 4))
    return result
