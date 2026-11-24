"""Lane R1 analysis (numpy only, runs anywhere on cells.json from chain/score_r1.py).

  python analyse_r1.py wave2 cells.json OUT_DIR      nested selection from the wave-1 coordinate arms -> arms_<order>.json for wave 2 (+ cheap single-move nested estimate)
  python analyse_r1.py report cells.json             contrasts of every arm vs ds_all (and cap250), people+orders 80 % intervals
Arm names in cells.json are `TAG:arm|d` (R1 jobs) or plain `stock`, `hardcb`, `ds_all` (lane P). Selection rule is the registered one in evidence/wave_20nov/laneR1_bigstack.md."""
import json
import re
import sys
from pathlib import Path

import numpy as np

KEYS = ("c1", "hb", "1", "2", "3", "4", "10", "9", "sc_fbraw", "sc_fb", "sc_sinc", "sc_fbl", "sc_rest")
EXTRAS = ("reve", "prat", "cbx", "xfam")
ORDERS = ("full0", "within0")


def parse(path):
    raw = json.loads(Path(path).read_text())["cells"]
    out = {}
    for o, arms in raw.items():
        out[o] = {}
        for k, v in arms.items():
            name = re.sub(r"\|d$", "", k.split(":", 1)[1] if ":" in k else k)
            if ":" in k:
                name = name if name != "base" else "base"
            people = sorted(v)
            out[o][name] = (np.array([v[p][0] for p in people]), np.array([v[p][1] for p in people]), people)
    return out


def ba(c, o, arm, mask):
    s, n, _ = c[o][arm]
    return 100 * s[mask].sum() / n[mask].sum()


def moves(c, orders):
    """name -> (kind, key, level) for every wave-1 coordinate arm present in all given orders."""
    out = {}
    for name in c["full0"]:
        m = re.fullmatch(r"m_(.+)_x([0-9.]+)", name)
        if m and all(name in c[o] for o in orders):
            out[name] = ("m", m.group(1), float(m.group(2)))
        m = re.fullmatch(r"a_(.+)_([0-9.]+)", name)
        if m and m.group(1) in EXTRAS and all(name in c[o] for o in orders):
            out[name] = ("a", m.group(1), float(m.group(2)))
    return out


def select(c, train, orders, shrink):
    """train: boolean mask over people. Per key the better level if its mean train delta vs base is > 0; extras from full0 only."""
    mult, add = {}, {}
    mv = moves(c, orders)
    for name, (kind, key, lvl) in mv.items():
        pass
    best = {}
    for name, (kind, key, lvl) in mv.items():
        orders_used = orders if kind == "m" else ("full0",)
        d = np.mean([ba(c, o, name, train) - ba(c, o, "base", train) for o in orders_used])
        if d > best.get((kind, key), (0, None))[0]:
            best[(kind, key)] = (d, lvl)
    for (kind, key), (d, lvl) in best.items():
        if kind == "m":
            mult[key] = float(lvl ** shrink)
        else:
            add[key] = float(lvl * shrink)
    return mult, add


def wave2(cells, outdir):
    c = parse(cells)
    people = [p.rsplit("/", 1)[-1] for p in c["full0"]["base"][2]]  # the chain keys people as P01..P10
    outdir = Path(outdir); outdir.mkdir(exist_ok=True)
    plan = {o: [] for o in ORDERS}
    single = {}
    for o in ORDERS:
        for src in ("self", "other", "both"):
            orders = {"self": (o,), "other": tuple(x for x in ORDERS if x != o), "both": ORDERS}[src]
            for shrink in (0.5, 1.0):
                W, Wadd = {}, {}
                for i, p in enumerate(people):
                    train = np.arange(len(people)) != i
                    m, a = select(c, train, orders, shrink)
                    W[p], Wadd[p] = m, (a if o == "full0" else {})  # extra banks exist for full0 only
                plan[o].append(dict(name=f"nest_{src}_s{shrink}", W=W, Wadd=Wadd))
    for o in ORDERS:
        for shrink in (0.5, 1.0):  # un-nested reference: selection on all ten people, both orders
            m, a = select(c, np.ones(len(people), bool), ORDERS, shrink)
            plan[o].append(dict(name=f"oracle_s{shrink}", mult=m, add=a))
    for o in ORDERS:
        Path(outdir / f"arms_{o}.json").write_text(json.dumps(plan[o], indent=1))
        print(o, len(plan[o]), "wave-2 arms")
    for s in (0.5, 1.0):
        print("oracle", s, select(c, np.ones(len(people), bool), ORDERS, s))


def contrast(c, a, b, orders, draw, extra=None):
    """paired people-resampled contrast a - b, orders averaged; returns pp, 80 % interval."""
    people = c[orders[0]]["base" if "base" in c[orders[0]] else "stock"][2]
    s = {x: np.mean([c[o][x][0] / c[o][x][1] for o in orders], axis=0) for x in (a, b)}
    n = c[orders[0]]["stock"][1]
    point = 100 * (np.mean([c[o][a][0] for o in orders], axis=0) - np.mean([c[o][b][0] for o in orders], axis=0)).sum() / n.sum()
    sa = np.mean([c[o][a][0] for o in orders], axis=0); sb = np.mean([c[o][b][0] for o in orders], axis=0)
    d = 100 * ((sa - sb)[draw].sum(1)) / n[draw].sum(1)
    return point, np.percentile(d, (10, 90))


def contrast_po(c, a, b, draws=10000):
    """people AND orders resampled jointly: each draw takes 10 people and, per draw, 2 orders with replacement from (full0, within0)."""
    rng = np.random.default_rng(461)
    pp = rng.integers(0, 10, (draws, 10)); oo = rng.integers(0, 2, (draws, 2))
    n = c["full0"]["stock"][1]
    d = np.stack([c[o][a][0] - c[o][b][0] for o in ORDERS])  # orders x people (sum of cell BA diffs)
    val = np.array([100 * d[oo[i]][:, pp[i]].mean(0).sum() / n[pp[i]].sum() for i in range(draws)])
    return np.percentile(val, (10, 90))


def nested_single(c):
    """cheap nested estimate: held-out person j scored at the single coordinate arm that is best on the other nine (both orders, extras full0 only), if positive."""
    mv = moves(c, ORDERS)
    n = c["full0"]["stock"][1]
    out = {}
    for o in ORDERS:
        tot = 0.0
        for j in range(10):
            train = np.arange(10) != j
            best, arm = 0.0, "base"
            for name, (kind, key, lvl) in mv.items():
                d = np.mean([ba(c, x, name, train) - ba(c, x, "base", train) for x in (ORDERS if kind == "m" else ("full0",))])
                if d > best:
                    best, arm = d, name
            tot += c[o][arm][0][j] - c[o]["base"][0][j]
        out[o] = 100 * tot / n.sum()
    return out


def report(cells):
    c = parse(cells)
    draw = np.random.default_rng(460).integers(0, 10, (10000, 10))
    arms = sorted(set(c["full0"]) & set(c["within0"]))
    rows = []
    for a in arms:
        if a in ("stock", "hardcb"):
            continue
        r = {}
        r["po"] = contrast_po(c, a, "ds_all") if a in c["within0"] else None
        for tag, orders in (("both", ORDERS), ("full0", ("full0",)), ("within0", ("within0",))):
            r[tag] = {ref: contrast(c, a, ref, orders, draw) for ref in ("ds_all", "stock")}
        rows.append((a, r))
    rows.sort(key=lambda x: -x[1]["both"]["ds_all"][0])
    print("nested single-move estimate vs base:", nested_single(c))
    print("arm | vs ds_all both [80% people] (full0/within0) [80% people+orders] | vs cap250 both")
    for a, r in rows:
        f = lambda x: f"{x[0]:+.3f} [{x[1][0]:+.2f},{x[1][1]:+.2f}]"
        print(f"{a:28s} {f(r['both']['ds_all'])} ({r['full0']['ds_all'][0]:+.2f}/{r['within0']['ds_all'][0]:+.2f}) [{r['po'][0]:+.2f},{r['po'][1]:+.2f}] | {f(r['both']['stock'])}")


if __name__ == "__main__":
    if sys.argv[1] == "wave2":
        wave2(sys.argv[2], sys.argv[3])
    else:
        report(sys.argv[2])
