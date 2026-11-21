"""Materialize exact gated identity from the native replay's recognition receipts."""
import argparse
from ast import literal_eval
from hashlib import file_digest
import json
from pathlib import Path
import re

import numpy as np

from returning_em_report import ARMS, ORDERS
from tapes import PEOPLE, ROSTERS, metadata, pinned, protocol, stream_ids

LINE = re.compile(r"^(new_P\d\d|ret_P\d\d) (nine25|ba_six) (718\d\d_[123]) (\d+) (\{.*\})$")


def log_records(root):
    records, digests = {}, {}
    paths = sorted(root.glob("lf-grow-held-65023988_*.out"))
    if len(paths) != 96:
        raise ValueError(f"expected 96 archived native job logs, read {len(paths)}")
    for path in paths:
        idx = int(path.stem.rsplit("_", 1)[1])
        first = 71800 + 2 * (idx // 12)
        with path.open("rb") as source:
            digests[path.name] = file_digest(source, "sha256").hexdigest()
        for line in path.read_text().splitlines():
            match = LINE.fullmatch(line)
            if match is None:
                continue
            fold, roster, key, count, receipt = match.groups()
            order = int(key.split("_")[0])
            if order not in (first, first + 1) or (fold, roster, key) in records:
                raise ValueError(f"ambiguous canonical recognition receipt in {path}")
            age = int(literal_eval(receipt)["recognition"])
            records[fold, roster, key] = (int(count), age, path.name)
    if len(records) != 576:
        raise ValueError(f"expected 576 archived stream receipts, read {len(records)}")
    return records, digests


def materialize(work, data, records):
    folds = [(mode, who) for mode, people in PEOPLE.items() for who in people]
    active, inactive = [], []
    for phase, orders in ORDERS.items():
        output = work / "returning_em" / phase
        output.mkdir(parents=True, exist_ok=True)
        for oi, first in enumerate(orders[::4]):
            chunk = (phase == "held") * 2 + oi
            for ri, roster in enumerate(ROSTERS):
                for fi, (mode, person) in enumerate(folds):
                    idx = 24 * chunk + 12 * ri + fi
                    fold = f"{mode}_{person}"
                    path = output / f"{roster}-{fold}-{first}.npz"
                    sessions = stream_ids(data, mode, person)
                    result, flags = {}, {}
                    for order in range(first, first + 4):
                        for stream, planned in protocol.plans(data, order, sessions).items():
                            key = f"{order}_{stream.split('|')[1]}"
                            count, age, log = records[fold, roster, key]
                            rows, labels, receipt = pinned(work / "leo", fold, roster, order, key)
                            np.testing.assert_array_equal(rows, planned)
                            if count != len(rows) or age < -1 or age >= len(rows):
                                raise ValueError(f"bad recognition age or row count in {log}")
                            if result and str(result["receipt"]) != receipt:
                                raise ValueError("fixture receipt changed")
                            result.update({"receipt": np.asarray(receipt), f"rows_{key}": rows,
                                           f"rec_{key}": np.asarray(-1), f"identity_{key}": np.asarray(""),
                                           f"gate_log_{key}": np.asarray(log)})
                            result.update({f"{arm}_{key}": labels for arm in ARMS})
                            if age >= 0:
                                flags[key] = age + 1
                    if flags:
                        active.append({"task": idx, "fold": fold, "roster": roster, "phase": phase,
                                       "recognition": flags})
                    else:
                        inactive.append(idx)
                        if not path.exists():
                            with path.open("xb") as f:
                                np.savez_compressed(f, **result)
    return {"active": active, "inactive": inactive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    data = metadata(args.data)
    records, digests = log_records(args.work / "em_archive_logs")
    results = materialize(args.work, data, records)
    with (args.work / "em_archive_manifest.json").open("x") as f:
        json.dump({**results, "logs": digests}, f, indent=2)
        f.write("\n")
    print(json.dumps({**results, "active_count": len(results["active"]),
                      "inactive_count": len(results["inactive"])}, indent=1))


if __name__ == "__main__":
    main()
