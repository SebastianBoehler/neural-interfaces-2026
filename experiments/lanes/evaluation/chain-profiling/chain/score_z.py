"""Lane Z scorer: stage map, leave-one-stage-out deltas and transmission tables from the ORDER_JOB.npz banks of final_z.py.
Official cell metric, ten reused held-new people, cells equal; people-resampled paired 80 % (10th/90th percentile, 10,000 draws).
Runs in the c25d worker environment: Z_JOB=score worker.sbatch score (Z_EXTRA="lev_a lev_b" adds job names). Output score_z.json and score_z.md."""
import json
import os
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES

WORK = Path(__file__).resolve().parent
ORDERS = ("full0", "within0")
JOBS = ("prof", "tx_slot_rand", "tx_slot_wrong", "tx_mix_rand", "tx_mix_wrong")
STAGES = ("slot", "mix", "rebal", "em", "tsem", "filtered", "terminal")
SYSTEMS = ("stock", "hardcb")
TX = ("slot_", "mix_")


def load_all(order, jobs):
    banks = {}
    for job in jobs:
        f = WORK / f"{order}_{job}.npz"
        if not f.is_file():
            continue
        assert json.loads((WORK / f"{order}_{job}.json").read_text())["passed"]
        with np.load(f, allow_pickle=False) as z:
            for k in z.files:
                if k in banks:
                    assert k.endswith("|base") or "|member|" in k or "|base|" in k or k in ("index", "delivery", "y"), f"duplicate arm array {k}"
                    np.testing.assert_array_equal(banks[k], z[k])  # the shared base arm must repeat bitwise across jobs
                else:
                    banks[k] = z[k]
    return banks


def main(extra_jobs=()):
    pin(METRIC, METRIC_HASH); pin(FEAT, HASHES[FEAT])
    metric = start_module("final_official_metric", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        y, subjects, sessions, interface = (z[k] for k in ("y", "subject", "session", "interface"))
    y = y.astype(int)
    groups = np.stack((subjects, sessions, interface), axis=1)
    people = sorted(set(subjects))
    draw = np.random.default_rng(470).integers(0, 10, (10000, 10))
    counts = np.array([len({tuple(g) for g in groups[subjects == p]}) for p in people], dtype=float)

    def person_sums(guess):
        return np.array([np.sum(metric.group_scores(metric.balanced_accuracy, y[subjects == p], guess[subjects == p], groups[subjects == p]))
                         for p in people])

    def pooled(v):
        return float(100 * v.sum() / counts.sum())

    def boot(v):
        return 100 * v[draw].sum(1) / counts[draw].sum(1)

    out, all_cs = {}, {}
    for order in ORDERS:
        banks = load_all(order, JOBS + tuple(extra_jobs))
        if "stock|base|terminal" not in banks:
            continue
        arms = sorted({k.rsplit("|", 1)[0] for k in banks if k.endswith("|terminal")})
        cs = {(a, s): person_sums(banks[f"{a}|{s}"].argmax(1)) for a in arms for s in STAGES}
        res = dict(stage_map={}, members={}, loso={}, tx={}, hard_frac={})
        for s in SYSTEMS:
            m = {st: pooled(cs[(f"{s}|base", st)]) for st in STAGES}
            m["by_person_terminal"] = dict(zip(people, (100 * cs[(f"{s}|base", "terminal")] / counts).tolist()))
            m["by_person_mix"] = dict(zip(people, (100 * cs[(f"{s}|base", "mix")] / counts).tolist()))
            res["stage_map"][s] = m
            res["members"][s] = {k.split("|")[-1]: pooled(person_sums(banks[k].argmax(1))) for k in banks if k.startswith(f"{s}|base|member|")}
        for a in arms:
            s, name = a.split("|")
            if name == "base":
                continue
            row = {}
            for st in STAGES:
                d = cs[(a, st)] - cs[(f"{s}|base", st)]
                row[st] = dict(pp=pooled(d), p10_90=np.percentile(boot(d), (10, 90)).tolist())
            res["tx" if name.startswith(TX) else "loso"][a] = row
        for s in SYSTEMS:
            mem = np.stack([banks[k].argmax(1) for k in banks if k.startswith(f"{s}|base|member|")])
            hard = (mem != y[None]).all(0)
            res["hard_frac"][s] = float(hard.mean())
            for a in res["tx"]:
                if not a.startswith(f"{s}|"):
                    continue
                gains = {}
                for st in STAGES:
                    d = (banks[f"{a}|{st}"].argmax(1) == y).astype(float) - (banks[f"{s}|base|{st}"].argmax(1) == y)
                    rw = banks[f"{s}|base|tsem"].argmax(1) != y  # windows the system's own raw stage gets wrong
                    gains[st] = dict(easy=float(100 * d[~hard].mean()), hard=float(100 * d[hard].mean()),
                                     raw_right=float(100 * d[~rw].mean()), raw_wrong=float(100 * d[rw].mean()))
                res["tx"][a]["windows"] = gains
        out[order] = res
        all_cs[order] = (cs, arms)
    both = {}
    if len(all_cs) == 2:
        (cf, af), (cw, aw) = all_cs["full0"], all_cs["within0"]
        for a in sorted(set(af) & set(aw)):
            s, name = a.split("|")
            row = {}
            for st in ("tsem", "terminal"):
                m = lambda c, k: c[(k, st)]  # noqa: E731
                d = 0.5 * (m(cf, a) + m(cw, a) - m(cf, f"{s}|base") - m(cw, f"{s}|base"))
                x = 0.5 * (m(cf, a) + m(cw, a) - m(cf, "stock|base") - m(cw, "stock|base"))
                row[st] = dict(vs_own_base=dict(pp=pooled(d), p10_90=np.percentile(boot(d), (10, 90)).tolist()),
                               vs_cap250=dict(pp=pooled(x), p10_90=np.percentile(boot(x), (10, 90)).tolist()))
            both[a] = row
        out["both_orders"] = both
    (WORK / "score_z.json").unlink(missing_ok=True)
    write_receipt(WORK / "score_z.json", scope="lane Z, ten reused held-new people", orders=out)
    lines = []
    for order, res in out.items():
        if order == "both_orders":
            lines.append("## both orders (mean of full0 and within0), vs own base | vs cap250 base")
            for a, row in res.items():
                if a.endswith("|base") or "|slot_" in a or "|mix_" in a:
                    continue
                t = row["terminal"]
                lines.append(f"BOTH {a}: terminal {t['vs_own_base']['pp']:+.2f} [{t['vs_own_base']['p10_90'][0]:+.2f},{t['vs_own_base']['p10_90'][1]:+.2f}] | vs cap250 {t['vs_cap250']['pp']:+.2f} [{t['vs_cap250']['p10_90'][0]:+.2f},{t['vs_cap250']['p10_90'][1]:+.2f}]")
            continue
        lines.append(f"## {order}")
        for s in SYSTEMS:
            m = res["stage_map"][s]
            lines.append(f"{s} stage BA: " + " ".join(f"{st} {m[st]:.2f}" for st in STAGES) + f" | hard windows {100 * res['hard_frac'][s]:.1f} %")
            lines.append(f"{s} members alone: " + " ".join(f"{k} {v:.1f}" for k, v in sorted(res["members"][s].items())))
        for a, row in res["loso"].items():
            r, t = row["tsem"], row["terminal"]
            lines.append(f"LOSO {a}: raw {r['pp']:+.2f} [{r['p10_90'][0]:+.2f},{r['p10_90'][1]:+.2f}] terminal {t['pp']:+.2f} [{t['p10_90'][0]:+.2f},{t['p10_90'][1]:+.2f}]")
        for a, row in res["tx"].items():
            pt = "slot" if "|slot_" in a else "mix"
            g = {st: row[st]["pp"] for st in STAGES}
            ratio = g["terminal"] / g[pt] if abs(g[pt]) > 1e-9 else float("nan")
            lines.append(f"TX {a}: " + " ".join(f"{st} {g[st]:+.2f}" for st in STAGES) + f" | T_terminal/{pt} {ratio:.2f}"
                         + f" | win easy {row['windows']['terminal']['easy']:+.2f} hard {row['windows']['terminal']['hard']:+.2f}"
                         + f" raw_right {row['windows']['terminal']['raw_right']:+.2f} raw_wrong {row['windows']['terminal']['raw_wrong']:+.2f}")
    (WORK / "score_z.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(os.environ.get("Z_EXTRA", "").split())
