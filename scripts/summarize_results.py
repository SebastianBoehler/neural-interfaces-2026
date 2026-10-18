"""Summarise experiment JSON lines into compact tracked evidence.

Usage: python scripts/summarize_results.py OUT.json RESULTS.jsonl [...]
Keeps the latest record per (name, protocol, seed), then reports per-fold
scores and, across seeds, the mean and sample SD of the ranking metric.
"""

import json
import sys
from collections import defaultdict
from statistics import mean, stdev


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    latest = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            latest[(r["name"], r["protocol"], r.get("seed", 0))] = r
    groups = defaultdict(list)
    for (name, proto, seed), r in sorted(latest.items()):
        groups[(name, proto)].append(r)
    summary = []
    for (name, proto), runs in sorted(groups.items()):
        scores = [r["balanced_accuracy_cell_mean"] for r in runs]
        summary.append({
            "name": name, "protocol": proto,
            "seeds": [r.get("seed", 0) for r in runs],
            "rev": sorted({r["rev"] for r in runs}),
            "cell_mean": round(mean(scores), 6),
            "cell_mean_sd": round(stdev(scores), 6) if len(scores) > 1 else None,
            "pooled_balanced_accuracy": round(
                mean(r["balanced_accuracy"] for r in runs), 6),
            "n_cells": runs[0]["n_cells"],
            "per_fold": {k: round(mean(r["per_fold"][k] for r in runs), 4)
                         for k in runs[0]["per_fold"]},
        })
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    for s in summary:
        sd = f"±{s['cell_mean_sd']:.3f}" if s["cell_mean_sd"] else ""
        print(f"{s['name']:20s} {s['protocol']:12s} n={len(s['seeds'])} "
              f"{s['cell_mean']:.4f}{sd}")


if __name__ == "__main__":
    main()
