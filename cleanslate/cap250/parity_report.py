"""Combine matched public replay receipts and enforce all-ten decision equality."""

import argparse
import json
from pathlib import Path

from source import PEOPLE


def summarize(source, out):
    reports = {person: json.loads((source / f"{person}.json").read_text()) for person in PEOPLE}
    references = {(row["sealed"], row["rebuilt_weights"]) for row in reports.values()}
    orders = {tuple(row["orders"]) for row in reports.values()}
    if len(references) != 1 or len(orders) != 1 or any(k != row["person"] for k, row in reports.items()):
        raise ValueError("public replay receipts used different ZIPs, weights, order lists or people")
    summary = {}
    for order in next(iter(orders)):
        results = [row["orders"][order] for row in reports.values()]
        windows, cells, changes = (sum(row[field] for row in results) for field in ("windows", "cells", "total"))
        if windows != 2655 or cells != 28 or sum(len(row["visits"]) for row in results) != 14:
            raise ValueError(f"public replay did not cover all 14 visits for {order}")
        summary[order] = dict(changed=changes, windows=windows, cells=cells,
                              sealed_ba=sum(row["sealed_ba_sum"] for row in results) / cells,
                              rebuilt_ba=sum(row["rebuilt_ba_sum"] for row in results) / cells)
    report = dict(summary=summary, people=reports)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if any(row["changed"] for row in summary.values()):
        raise SystemExit("fresh all-ten replay changed public decisions relative to sealed stock")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.out)
