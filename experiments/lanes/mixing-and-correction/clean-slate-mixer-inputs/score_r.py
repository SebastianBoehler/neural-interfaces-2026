"""Revisit scorer (local, numpy): ORDER_rev.npz banks of revisit/chain/final_r.py -> stage map and paired deltas vs stock|base.

python revisit/score_r.py full0_rev.npz within0_rev.npz [--out revisit_score.json]
Official cell-mean BA (submodule benchmark_utils.metrics), ten held-new people, cells equal. Paired 80 % interval: 10,000 draws
resampling people (cell-weighted) and orders (with replacement) jointly; per-order intervals resample people only."""
import argparse
import json

import numpy as np

from offline import G, M, PEOPLE, SUBJ, Y

STAGES = ("cap_mix", "mix", "rebal", "em", "tsem", "filtered", "terminal")
COUNTS = np.array([len({tuple(g) for g in G[SUBJ == p]}) for p in PEOPLE], float)


def person_sums(post):
    g = post.argmax(1)
    return np.array([np.sum(M.group_scores(M.balanced_accuracy, Y[SUBJ == p], g[SUBJ == p], G[SUBJ == p])) for p in PEOPLE])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("banks", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = np.random.default_rng(470)
    banks = [np.load(b) for b in a.banks]
    arms = sorted({k.rsplit("|", 1)[0] for k in banks[0].files if k.endswith("|terminal")})
    for z in banks:
        np.testing.assert_array_equal(z["y"], Y)
    sums = {(o, arm, s): person_sums(z[f"{arm}|{s}"].astype(float)) for o, z in enumerate(banks) for arm in arms for s in STAGES}
    no = len(banks)
    dp, do = rng.integers(0, 10, (10000, 10)), rng.integers(0, no, (10000, no))
    res = {}
    for arm in arms:
        r = dict(stage={s: [float(100 * sums[(o, arm, s)].sum() / COUNTS.sum()) for o in range(no)] for s in STAGES})
        if arm != "stock|base":
            d = np.stack([sums[(o, arm, "terminal")] - sums[(o, "stock|base", "terminal")] for o in range(no)])  # orders x people
            pooled = 100 * d.sum() / (no * COUNTS.sum())
            boot = 100 * np.array([d[do[i]][:, dp[i]].sum() / (no * COUNTS[dp[i]].sum()) for i in range(10000)])
            per = [[float(100 * d[o].sum() / COUNTS.sum()), *np.percentile(100 * d[o][dp].sum(1) / COUNTS[dp].sum(1), [10, 90]).tolist()] for o in range(no)]
            r.update(delta=float(pooled), lo=float(np.percentile(boot, 10)), hi=float(np.percentile(boot, 90)), per_order=per,
                     P04=float(100 * d[:, PEOPLE.index("P04")].mean() / COUNTS[PEOPLE.index("P04")]),
                     P10=float(100 * d[:, PEOPLE.index("P10")].mean() / COUNTS[PEOPLE.index("P10")]),
                     people_pos=int(((d.sum(0)) > 0).sum()))
        res[arm] = r
    print("| arm | " + " | ".join(STAGES) + " | Δ terminal [80 %] | per order | P04 | P10 | people + |")
    print("|---" * (len(STAGES) + 6) + "|")
    for arm, r in res.items():
        st = " | ".join("/".join(f"{v:.1f}" for v in r["stage"][s]) for s in STAGES)
        if "delta" in r:
            po = " ; ".join(f"{p[0]:+.2f} [{p[1]:+.2f},{p[2]:+.2f}]" for p in r["per_order"])
            print(f"| {arm} | {st} | {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {po} | {r['P04']:+.1f} | {r['P10']:+.1f} | {r['people_pos']}/10 |")
        else:
            print(f"| {arm} | {st} | — | — | — | — | — |")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
