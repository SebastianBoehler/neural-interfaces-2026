"""Lane S2 analysis (numpy only; cells.json from chain/score_s2.py). Rule registered in evidence/wave_20nov/laneS2_session_stages.md.

  python analyse_s2.py wave2 cells.json OUT_DIR   nested + un-nested selection from the wave-1 moves -> arms_<order>_<k>.json (wave-2 chain jobs, file:arms...json)
  python analyse_s2.py report cells.json          wave-1 single moves and wave-2 forms vs base / hardcb / ds_all / cap250 (people 80 % and people+orders 80 %)
Arms in cells.json: `sys|name` (S2 jobs) and plain stock, hardcb, ds_all (lane P)."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "chain"))
from consts_s2 import DEF, MOVES  # noqa: E402

ORDERS = ("full0", "within0")
SYSTEMS = ("hardcb", "ds_all")
SCOPES = {"rebal": ("rebal",), "em": ("em",), "tsem": ("tsem",), "tail": ("csp", "count"), "all": ("rebal", "em", "tsem", "csp", "count")}
SHIP = {**{s: dict(c) for s, c in DEF.items()}}
SHIP["csp"] = dict(on=1, k=20, q=0.5, cap=0.10)
INTS = {("rebal", "W"), ("em", "n0"), ("em", "m0"), ("em", "nmin"), ("tsem", "m0"), ("tsem", "n0"), ("tsem", "nmin"), ("csp", "k")}


def parse(path):
    raw = json.loads(Path(path).read_text())["cells"]
    c = {}
    for o, arms in raw.items():
        c[o] = {}
        for k, v in arms.items():
            ps = sorted(v)
            c[o][k] = (np.array([v[p][0] for p in ps]), np.array([v[p][1] for p in ps]), ps)
    return c


def ba(c, o, arm, mask):
    s, n, _ = c[o][arm]
    return 100 * s[mask].sum() / n[mask].sum()


def value(stage, const, level, s):
    v = SHIP[stage][const] + s * (level - SHIP[stage][const])
    return int(round(v)) if (stage, const) in INTS else float(v)


def select(c, system, train, orders, shrink, stages):
    """per constant the better level (mean train delta over orders > 0) -> {stage: {const: value}}; on/off flags only at shrink 1."""
    out = {}
    for st in stages:
        for const, levels in MOVES[st].items():
            best = (0.0, None)
            for lv in levels:
                name = f"{system}|{st}.{const}={lv}"
                if not all(name in c[o] for o in orders):
                    continue
                d = np.mean([ba(c, o, name, train) - ba(c, o, f"{system}|base", train) for o in orders])
                if d > best[0]:
                    best = (d, lv)
            if best[1] is None or (const == "on" and shrink < 1):
                continue
            out.setdefault(st, {})[const] = best[1] if const == "on" else value(st, const, best[1], shrink)
    return out


def wave2(cells, outdir, parts=8):
    c = parse(cells)
    people = c["full0"]["hardcb|base"][2]
    short = [p.rsplit("/", 1)[-1] for p in people]
    plan = {o: [] for o in ORDERS}
    for o in ORDERS:
        for sysm in SYSTEMS:
            for scope, stages in SCOPES.items():
                for shrink in (1.0, 0.5):
                    for src in ("self", "other", "both"):
                        srco = {"self": (o,), "other": tuple(x for x in ORDERS if x != o), "both": ORDERS}[src]
                        P = {short[i]: select(c, sysm, np.arange(10) != i, srco, shrink, stages) for i in range(10)}
                        plan[o].append(dict(name=f"nest_{src}_{scope}_s{shrink}", sys=sysm, set={}, P=P))
                    g = select(c, sysm, np.ones(10, bool), ORDERS, shrink, stages)
                    plan[o].append(dict(name=f"oracle_{scope}_s{shrink}", sys=sysm, set=g))
                    print(o, sysm, scope, shrink, "oracle:", json.dumps(g))
    outdir = Path(outdir); outdir.mkdir(exist_ok=True)
    for o in ORDERS:
        for i in range(parts):
            arms = [a for j, a in enumerate(plan[o]) if j % parts == i]
            (outdir / f"arms_{o}_{i}.json").write_text(json.dumps(arms))
        print(o, len(plan[o]), "wave-2 arms in", parts, "files")


def wave3(cells, outdir, top=4):
    """candidate whole-config arms: all subsets (size 2..top) of the top positive single moves per system (both-order mean delta > 0, different constants)."""
    import itertools
    c = parse(cells); ones = np.ones(10, bool)
    plan = []
    for sysm in SYSTEMS:
        d = {}
        for a in c["full0"]:
            m = re.fullmatch(re.escape(sysm) + r"\|(\w+)\.(\w+)=(.+)", a)
            if m and a in c["within0"]:
                d[(m.group(1), m.group(2), float(m.group(3)))] = np.mean([ba(c, o, a, ones) - ba(c, o, f"{sysm}|base", ones) for o in ORDERS])
        pos = sorted((k for k, v in d.items() if v > 0), key=lambda k: -d[k])[:top]
        print(sysm, "top positive moves", [(k, round(d[k], 2)) for k in pos])
        for r in range(2, len(pos) + 1):
            for sub in itertools.combinations(pos, r):
                if len({(k[0], k[1]) for k in sub}) < r:
                    continue
                st = {}
                for (a_, b_, v) in sub:
                    st.setdefault(a_, {})[b_] = int(v) if (a_, b_) in INTS else (int(v) if b_ == "on" else v)
                plan.append(dict(name="combo_" + "+".join(f"{a_}.{b_}={v}" for a_, b_, v in sub), sys=sysm, set=st))
    outdir = Path(outdir); outdir.mkdir(exist_ok=True)
    (outdir / "arms_w3.json").write_text(json.dumps(plan))
    print(len(plan), "wave-3 arms")


def pick(c, system, o, src, pool):
    """nested whole-arm selection: held-out j takes the pool arm with the best mean train delta (people != j, orders per src) if > 0 else base; returns (pp vs base for order o, chosen arms)."""
    srco = {"self": (o,), "other": tuple(x for x in ORDERS if x != o), "both": ORDERS}[src]
    n = c[o][f"{system}|base"][1]; tot = 0.0; chosen = []
    for j in range(10):
        train = np.arange(10) != j
        best, arm = 0.0, f"{system}|base"
        for a in pool:
            d = np.mean([ba(c, x, a, train) - ba(c, x, f"{system}|base", train) for x in srco])
            if d > best:
                best, arm = d, a
        tot += c[o][arm][0][j] - c[o][f"{system}|base"][0][j]; chosen.append(arm.split("|")[1])
    return 100 * tot / n.sum(), chosen


def contrast(c, a, b, orders, draw):
    sa = np.mean([c[o][a][0] for o in orders], axis=0); sb = np.mean([c[o][b][0] for o in orders], axis=0)
    n = c[orders[0]][a][1]
    return 100 * (sa - sb).sum() / n.sum(), np.percentile(100 * (sa - sb)[draw].sum(1) / n[draw].sum(1), (10, 90))


def contrast_po(c, a, b, draws=10000):
    """people and orders resampled jointly (10 people, 2 orders with replacement per draw)."""
    rng = np.random.default_rng(461)
    pp = rng.integers(0, 10, (draws, 10)); oo = rng.integers(0, 2, (draws, 2))
    n = c["full0"][a][1]
    d = np.stack([c[o][a][0] - c[o][b][0] for o in ORDERS])
    return np.percentile([100 * d[oo[i]][:, pp[i]].mean(0).sum() / n[pp[i]].sum() for i in range(draws)], (10, 90))


def person_delta(c, a, b, who):
    i = [p.rsplit("/", 1)[-1] for p in c["full0"][a][2]].index(who)
    return np.mean([100 * (c[o][a][0][i] - c[o][b][0][i]) / c[o][a][1][i] for o in ORDERS])


def report(cells):
    c = parse(cells)
    draw = np.random.default_rng(460).integers(0, 10, (10000, 10))
    f = lambda x: f"{x[0]:+.2f} [{x[1][0]:+.2f},{x[1][1]:+.2f}]"
    for sysm in SYSTEMS:
        base = f"{sysm}|base"
        print(f"\n## {sysm}: base {np.mean([ba(c, o, base, np.ones(10, bool)) for o in ORDERS]):.2f} (full0 {ba(c, 'full0', base, np.ones(10, bool)):.2f} / within0 {ba(c, 'within0', base, np.ones(10, bool)):.2f})")
        rows = []
        for a in sorted(k for k in c["full0"] if k.startswith(sysm + "|") and k != base and k in c["within0"]):
            r = {t: contrast(c, a, base, os_, draw) for t, os_ in (("both", ORDERS), ("full0", ("full0",)), ("within0", ("within0",)))}
            rows.append((a, r, contrast_po(c, a, base)))
        rows.sort(key=lambda x: (not x[0].split("|")[1].startswith(("nest", "oracle")), -x[1]["both"][0]))
        print("arm | vs own base both [people 80%] (full0/within0) [people+orders 80%] | P04 P10 | vs hardcb | vs ds_all | vs cap250")
        for a, r, po in rows:
            tail = " | ".join(f(contrast(c, a, ref, ORDERS, draw)) if ref in c["full0"] else "-" for ref in ("hardcb", "ds_all", "stock"))
            print(f"{a.split('|')[1]:28s} {f(r['both'])} ({r['full0'][0]:+.2f}/{r['within0'][0]:+.2f}) [{po[0]:+.2f},{po[1]:+.2f}] | "
                  f"{person_delta(c, a, base, 'P04'):+.2f} {person_delta(c, a, base, 'P10'):+.2f} | {tail}")


def report_pick(cells):
    c = parse(cells)
    for sysm in SYSTEMS:
        pool = sorted(k for k in c["full0"] if k.startswith(sysm + "|") and not k.split("|")[1].startswith(("nest", "oracle", "base")) and k in c["within0"])
        for src in ("self", "other", "both"):
            r = {o: pick(c, sysm, o, src, pool) for o in ORDERS}
            print(f"{sysm} nested pick over {len(pool)} whole-config arms, selection {src}: full0 {r['full0'][0]:+.2f} within0 {r['within0'][0]:+.2f} mean {np.mean([r[o][0] for o in ORDERS]):+.2f}; "
                  f"chosen (full0) {sorted(set(r['full0'][1]))}")
        ones = np.ones(10, bool)
        best = max(pool, key=lambda a: np.mean([ba(c, o, a, ones) - ba(c, o, f"{sysm}|base", ones) for o in ORDERS]))
        print(f"{sysm} un-nested best single arm {best}: {np.mean([ba(c, o, best, ones) - ba(c, o, f'{sysm}|base', ones) for o in ORDERS]):+.2f}")


if __name__ == "__main__":
    {"wave2": lambda: wave2(sys.argv[2], sys.argv[3]), "wave3": lambda: wave3(sys.argv[2], sys.argv[3]), "pick": lambda: report_pick(sys.argv[2])}.get(sys.argv[1], lambda: report(sys.argv[2]))()
