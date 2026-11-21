"""Extend the frozen ba_six capture without scoring or running recognition oracles."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from capture import DATA_SHA, V, capture, fixture_assets, load, open_fold, pin, replay, sessions

ROSTER = "ba_six"
STARTS = (71804, 71806, 71812, 71814)
FIELDS = ("hard", "team", "own", "count", "csp", "csp_active", "manas", "terminal", "label", "recognised")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("fold")
    parser.add_argument("first", type=int, choices=STARTS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or V.digest(args.data) != DATA_SHA:
        raise ValueError("CUDA and the signed PROTEUS export are required")
    data = V.D.load(args.data)
    signals = torch.from_numpy(data["X"])
    pkg, receipts = open_fold(args.fixtures, args.fold, "ba-confirm")
    receipt = receipts[ROSTER]
    fixture_assets(pkg / ROSTER, receipt)
    model, _ = load(data, pkg, ROSTER, receipt)
    result = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    for order in (args.first, args.first + 1):
        for stream, rows in V.D.plans(data, order, sessions(data, args.fold)).items():
            captured, items = capture(model, signals, rows)
            pin(args.fixtures, args.fold, ROSTER, order, stream, rows, captured["label"], receipt)
            np.testing.assert_array_equal(replay(model, items, "w25")[0], captured["label"])
            if args.smoke:
                prefix, _ = capture(model, signals, rows[:32])
                for name in FIELDS:
                    if np.issubdtype(prefix[name].dtype, np.floating):
                        np.testing.assert_allclose(prefix[name], captured[name][:32], rtol=0, atol=2e-5)
                    else:
                        np.testing.assert_array_equal(prefix[name], captured[name][:32])
            key = f"{order}_{stream.split('|')[1]}"
            result[f"rows_{key}"] = rows
            result.update({f"{name}_{key}": captured[name] for name in FIELDS})
            print(f"PASS {args.fold}/{key}, {len(rows)} pinned labels and exact tape replay", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out) as tmp:
        partial = Path(tmp) / f"{ROSTER}-{args.fold}-{args.first}.npz"
        np.savez_compressed(partial, **result)
        partial.rename(args.out / partial.name)


if __name__ == "__main__":
    main()
