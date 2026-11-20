"""Verify paired replay receipts and score with the clean-slate harness."""
import argparse
import json
from pathlib import Path

import numpy as np

from base import D, roster
from external import CORPORA
from package import ARMS
from score import collect


def verify(root, data, smoke):
    people = D.PEOPLE[:1] if smoke else D.PEOPLE
    orders = D.DEFAULT_ORDERS[:1] if smoke else D.DEFAULT_ORDERS
    timings = {arm: [0.0, 0] for arm in ARMS}
    inputs = {}
    for held in people:
        fold = json.loads((root / "packages" / held / "fit.json").read_text())
        inner = roster((held,))[:1] if smoke else roster((held,))
        if (fold["held"] != held or fold["outer"] != roster((held,))
                or fold["inner"] != {e: roster((held, e)) for e in inner}
                or set(fold["external"]) != set(("ds007221g",) if smoke else CORPORA)):
            raise ValueError(f"bad training receipt for {held}")
        inputs[held] = {k: fold[k] for k in ("weights_sha256", "context_sha256", "solver_sha256")}
        streams = sorted(set(data["stream"][data["person"] == held]))
        plans = {order: D.plans(data, order, streams) for order in orders}
        if smoke:
            plans[orders[0]] = {streams[0]: plans[orders[0]][streams[0]][:18]}
        for arm in ARMS:
            with np.load(root / arm / f"pred_new_{held}.npz", allow_pickle=False) as z:
                expected = set()
                for order, plan in plans.items():
                    for s, rows in plan.items():
                        suffix = f"{order}_{s.split('|')[1]}"
                        keys = [f"{prefix}_{suffix}" for prefix in ("rows", "labels", "secs")]
                        expected.update(keys)
                        indices, labels, secs = (z[key] for key in keys)
                        np.testing.assert_array_equal(indices, rows)
                        if (labels.shape != rows.shape or not np.all((0 <= labels) & (labels <= 2))
                                or not np.isfinite(secs) or secs <= 0):
                            raise ValueError(f"invalid prediction or timing for {held} {arm} {s} {order}")
                        timings[arm][0] += float(secs)
                        timings[arm][1] += len(rows)
                if set(z.files) != expected:
                    raise ValueError(f"extra replay rows in {held} {arm}")
    return {arm: 1000 * secs / n for arm, (secs, n) in timings.items()}, inputs


def interval(draws, level):
    tail = 50 * (1 - level)
    return np.percentile(draws, [tail, 100 - tail]).round(4).tolist()


def summarize(tables):
    rng = np.random.default_rng(1675)
    draws = 10_000
    people = rng.integers(len(D.PEOPLE), size=(draws, len(D.PEOPLE)))
    orders = rng.integers(len(D.DEFAULT_ORDERS), size=(*people.shape, len(D.DEFAULT_ORDERS)))
    totals, boot, rows = {}, {}, {}
    for arm, cells in tables.items():
        score = np.empty((len(D.PEOPLE), len(D.DEFAULT_ORDERS)))
        count = np.empty_like(score)
        for i, p in enumerate(D.PEOPLE):
            for j, order in enumerate(D.DEFAULT_ORDERS):
                vals = [c["ba"] for c in cells if c["person"] == p and c["order"] == order]
                score[i, j], count[i, j] = sum(vals), len(vals)
        if not (count.sum(axis=0) == 28).all() or not count.all():
            raise ValueError(f"incomplete held-new cell scores for {arm}")
        selection = (people[:, :, None], orders)
        boot[arm] = score[selection].sum(axis=(1, 2)) / count[selection].sum(axis=(1, 2))
        totals[arm] = float(score.sum() / count.sum())
        rows[arm] = dict(ba=totals[arm], i80=interval(boot[arm], 0.8), i95=interval(boot[arm], 0.95),
                         by_order=dict(zip(D.DEFAULT_ORDERS, (score.sum(0) / count.sum(0)).tolist())),
                         by_person=dict(zip(D.PEOPLE, (score.sum(1) / count.sum(1)).tolist())))
    diffs = {}
    for a, b in (("context", "blank"), ("context", "a2h"), ("a2h", "a1")):
        gain = boot[a] - boot[b]
        diffs[f"{a}-{b}"] = dict(pp=totals[a] - totals[b], i80=interval(gain, 0.8), i95=interval(gain, 0.95))
    return rows, diffs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    data = D.load()
    timing, inputs = verify(args.root, data, args.smoke)
    if args.smoke:
        print("PASS unscored paired replay receipt and pinned-order check", flush=True)
        return
    tables = {arm: collect(args.root / arm, data)[0] for arm in ARMS}
    scores, comparisons = summarize(tables)
    output = dict(scores=scores, comparisons=comparisons, ms_per_window=timing, inputs=inputs)
    text = json.dumps(output, indent=2)
    (args.root / "report.json").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
