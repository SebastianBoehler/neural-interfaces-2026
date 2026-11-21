"""Re-tap only recognised streams and authenticate the full em_half output."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from returning_em_worker import DATA_SHA, V, fixture_assets, load, modified, open_fold, pin, taped
from final_review_emissions import COUNT_DOSES, shrink_count

ROSTER = "ba_six"


def replay_count(model, items, strength):
    count_filter = type(model.cf)
    native = count_filter.step
    age = 0

    def step(owner, x, p):
        nonlocal age
        q = native(owner, x, p)
        out = shrink_count(p, q, items[age].recognised, strength)
        age += 1
        return out

    with patch.object(count_filter, "step", step):
        result = modified(model, items, "half")
    if age != len(items):
        raise ValueError("CountFilter did not consume every captured window")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "archive", "manifest", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("fold")
    parser.add_argument("phase", choices=("fit", "held"))
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    if not torch.cuda.is_available() or V.digest(a.data) != DATA_SHA:
        raise ValueError("CUDA and signed PROTEUS data required")
    data = V.D.load(a.data)
    signals = torch.from_numpy(data["X"])
    start = 71800 if a.phase == "fit" else 71808
    active = {key: age for block in json.loads(a.manifest.read_text())["active"]
              if block["roster"] == ROSTER and block["fold"] == a.fold and block["phase"] == a.phase
              for key, age in block["recognition"].items()
              if start <= int(key.split("_")[0]) < start + 4}
    if not active:
        raise ValueError("requested slice has no canonical recognition")
    if a.smoke:
        key = min(active)
        active = {key: active[key]}
    pkg, receipts = open_fold(a.fixtures, a.fold, "ba-confirm")
    receipt = receipts[ROSTER]
    fixture_assets(pkg / ROSTER, receipt)
    model, _ = load(data, pkg, ROSTER, receipt)
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    expected_file = a.archive / a.phase / f"{ROSTER}-{a.fold}-{start}.npz"
    with np.load(expected_file, allow_pickle=False) as expected:
        if str(expected["receipt"]) != str(payload["receipt"]):
            raise ValueError("em_half archive used different fold assets")
        for key, age in sorted(active.items()):
            order, visit = key.split("_")
            stream = f"{a.fold.split('_')[1]}|{visit}"
            rows = V.D.plans(data, int(order), [stream])[stream]
            items = taped(model, signals, rows)
            pin(a.fixtures, a.fold, ROSTER, int(order), stream, rows,
                np.array([item.label for item in items]), receipt)
            flags = np.asarray([item.recognised for item in items])
            np.testing.assert_array_equal(flags, np.arange(len(rows)) >= age - 1)
            np.testing.assert_array_equal(rows, expected[f"rows_{key}"])
            native, scores = replay_count(model, items, 0.0)
            np.testing.assert_array_equal(native, expected[f"half_{key}"])
            payload.update({f"rows_{key}": rows, f"terminal_{key}": scores,
                            f"recognised_{key}": flags, f"label_{key}": native})
            for strength in COUNT_DOSES[1:]:
                labels, q = replay_count(model, items, strength)
                np.testing.assert_array_equal(q[~flags], scores[~flags])
                if a.smoke:
                    cut_labels, cut_q = replay_count(model, items[:84], strength)
                    np.testing.assert_array_equal(cut_labels, labels[:84])
                    np.testing.assert_array_equal(cut_q, q[:84])
                payload[f"count{strength:g}_{key}"] = labels
            print(f"PASS {a.fold} {key}: {len(rows)} pinned native and em_half labels; latch {age}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out) as temp:
        target = Path(temp) / f"{a.fold}-{a.phase}.npz"
        np.savez_compressed(target, **payload)
        target.rename(a.out / target.name)


if __name__ == "__main__":
    main()
