"""Lane P3: dsu_all vs hardcb vs cap250 per cell on one comparator, from stored lane P2 outputs.
usage: laneP3_compare.py P3_DIR DATA_NPZ OUT_JSON   (P3_DIR holds full0_P.npz, within0_P.npz, ret_u/<order>_<P>_<s>/*.npz)
Fails loudly on a missing file, an arm that is absent, or a cell mismatch."""
import json
import sys
from pathlib import Path

import numpy as np

D, DATA, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ORD = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
SCHEME = {"within": ORD[:4], "full": ORD[4:6], "runs": ORD[6:]}
z = np.load(DATA)
y, sub, ses, itf = z["y"], z["subject"], z["session"], z["interface"]
rng = np.random.default_rng(2103)


def cell_ba(rows, pred):
    """official-style: per (subject, session, interface) cell balanced accuracy; rows index the data, pred aligned to rows"""
    out = {}
    for key in sorted(set(zip(sub[rows], ses[rows], itf[rows]))):
        m = (sub[rows] == key[0]) & (ses[rows] == key[1]) & (itf[rows] == key[2])
        yy, pp = y[rows][m], pred[m]
        out[key] = 100 * np.mean([(pp[yy == c] == c).mean() for c in range(3)])
    return out


def ceiling(rows, base, other):
    """per cell: gain if every window where other != base and base is wrong were right (and nothing else), class-recall weighted; floor symmetric"""
    res = {}
    for key in sorted(set(zip(sub[rows], ses[rows], itf[rows]))):
        m = (sub[rows] == key[0]) & (ses[rows] == key[1]) & (itf[rows] == key[2])
        yy, b, o = y[rows][m], base[m], other[m]
        ch = b != o
        up = sum(1 / (3 * (yy == c).sum()) for c in yy[ch & (b != yy)])
        dn = sum(1 / (3 * (yy == c).sum()) for c in yy[ch & (b == yy)])
        res[key] = dict(n=int(m.sum()), changed=int(ch.sum()), fixed=int((ch & (o == yy)).sum()), broken=int((ch & (b == yy)).sum()),
                        ceiling_pp=100 * up, floor_pp=-100 * dn)
    return res


# ---- held-new chain (rows = all 2655, terminal argmax) ----
new = {}
for o in ("full0", "within0"):
    c = np.load(D / f"{o}_P.npz")
    assert (c["index"] == np.arange(len(y))).all()
    new[o] = {a: c[f"{a}|terminal"].argmax(-1) for a in ("stock", "hardcb", "ds_all")}
allrows = np.arange(len(y))
newcells = {o: {a: cell_ba(allrows, new[o][a]) for a in new[o]} for o in new}
keys = sorted(newcells["full0"]["stock"])
people = sorted(set(k[0] for k in keys))
CON = (("dsu_all-hardcb", "ds_all", "hardcb"), ("dsu_all-cap250", "ds_all", "stock"), ("hardcb-cap250", "hardcb", "stock"))
res = dict(held_new={}, returning={}, notes=[])
for tag, ords in (("full0", ("full0",)), ("within0", ("within0",)), ("both", ("full0", "within0"))):
    pc = {}  # person -> contrast -> mean over orders of mean over that person's cells
    for p in people:
        ks = [k for k in keys if k[0] == p]
        pc[p] = {n: np.mean([np.mean([newcells[o][a][k] - newcells[o][b][k] for k in ks]) for o in ords]) for n, a, b in CON}
    # equal cells: weight each person by n cells
    w = np.array([sum(1 for k in keys if k[0] == p) for p in people], float)
    r = {}
    for n, _, _ in CON:
        v = np.array([pc[p][n] for p in people])
        boot = [np.sum(w[i] * v[i]) / w[i].sum() for i in (rng.integers(0, len(people), len(people)) for _ in range(10000))]
        r[n] = dict(pp=float(np.sum(w * v) / w.sum()), people80=np.percentile(boot, (10, 90)).tolist(),
                    per_person={p: float(pc[p][n]) for p in people})
    res["held_new"][tag] = r
cells_new = {o: {n: {k: newcells[o][a][k] - newcells[o][b][k] for k in keys} for n, a, b in CON} for o in new}
res["held_new_cells"] = {o: {n: {"|".join(k): v for k, v in d.items()} for n, d in cells_new[o].items()} for o in new}
res["held_new_changed"] = {o: {"|".join(k): v for k, v in ceiling(allrows, new[o]["hardcb"], new[o]["ds_all"]).items()} for o in new}

# ---- returning P04/P10 visits 2,3 ----
RA = ("stock", "hardcb", "ftlite5", "ftdsu_all")
ret = {}
for o in ORD:
    for p in ("P04", "P10"):
        for s in (2, 3):
            f = D / "ret_u" / f"{o}_{p}_{s}" / f"{o}_{p}_{s}.npz"
            assert f.exists(), f
            q = np.load(f)
            ret[(o, p, s)] = (q["rows"], {a: q[f"{a}|labels"] for a in RA})
rcell = {}
for (o, p, s), (rows, lab) in ret.items():
    for a in RA:
        for k, v in cell_ba(rows, lab[a]).items():
            rcell[(o, a, k)] = v
rkeys = sorted(set(k for (_, _, k) in rcell))
RCON = (("dsu_all-hardcb", "ftdsu_all", "hardcb"), ("dsu_all-cap250", "ftdsu_all", "stock"), ("hardcb-cap250", "hardcb", "stock"),
        ("lite5-hardcb", "ftlite5", "hardcb"), ("dsu_all-lite5", "ftdsu_all", "ftlite5"))
for sch, ords in list(SCHEME.items()) + [("all8", ORD)]:
    r = {}
    for n, a, b in RCON:
        per = {p: np.mean([[rcell[(o, a, k)] - rcell[(o, b, k)] for k in rkeys if k[0].endswith(p)] for o in ords]) for p in ("P04", "P10")}
        eq = np.mean([rcell[(o, a, k)] - rcell[(o, b, k)] for o in ords for k in rkeys])
        # nested bootstrap: person draw then order draw within person
        def draw():
            ps = rng.integers(0, 2, 2)
            return np.mean([np.mean([np.mean([rcell[(o, a, k)] - rcell[(o, b, k)] for k in rkeys if k[0].endswith(("P04", "P10")[p])])
                                     for o in rng.choice(ords, len(ords))]) for p in ps])
        bo = [draw() for _ in range(2000)]
        w2080 = 0.2 * per["P04"] + 0.8 * per["P10"]
        r[n] = dict(equal_cell=float(eq), P04=float(per["P04"]), P10=float(per["P10"]), w20_80_UNVERIFIED=float(w2080),
                    nested_people_orders80=np.percentile(bo, (10, 90)).tolist(),
                    orders_neg=int(sum(np.mean([rcell[(o, a, k)] - rcell[(o, b, k)] for k in rkeys]) < 0 for o in ords)), n_orders=len(ords))
    res["returning"][sch] = r
res["returning_cells_all8"] = {"|".join(k): {n: float(np.mean([rcell[(o, a, k)] - rcell[(o, b, k)] for o in ORD])) for n, a, b in RCON} for k in rkeys}
# changed predictions dsu_all vs hardcb (and vs lite5) on returning cells
chg = {}
for (o, p, s), (rows, lab) in ret.items():
    for nm, base in (("vs_hardcb", "hardcb"), ("vs_lite5", "ftlite5")):
        for k, v in ceiling(rows, lab[base], lab["ftdsu_all"]).items():
            d = chg.setdefault(nm, {}).setdefault("|".join(k), []); d.append(v)
res["returning_changed"] = {nm: {k: {f: float(np.mean([x[f] for x in v])) for f in v[0]} for k, v in d.items()} for nm, d in chg.items()}
tot = lambda nm, f: float(np.mean([vv[f] for vv in res["returning_changed"][nm].values()]))
res["returning_changed_mean"] = {nm: {f: tot(nm, f) for f in ("n", "changed", "fixed", "broken", "ceiling_pp", "floor_pp")} for nm in res["returning_changed"]}
Path(OUT).write_text(json.dumps(res, indent=1, sort_keys=True))
print(json.dumps({k: res[k] for k in ("held_new", "returning", "returning_changed_mean")}, indent=1)[:9000])
