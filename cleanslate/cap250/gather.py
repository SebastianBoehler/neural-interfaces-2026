"""Join disjoint cluster prediction shards and score all 12 official public folds."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from source import PEOPLE, ROOT, default_data

sys.path.insert(0, str(ROOT / "cleanslate/harness"))
import data as harness

ORDERS = (71000, 71001, 71002, 71003)
FOLDS = {("new", person) for person in PEOPLE} | {("ret", person) for person in ("P04", "P10")}


def check_prediction(data, path, mode, person):
    sessions = sorted(set(data["session"][data["person"] == person]))
    if mode == "ret":
        sessions = ["2", "3"]
    streams = [f"{person}|{session}" for session in sessions]
    with np.load(path, allow_pickle=False) as rows:
        keys = {f"{kind}_{order}_{visit}" for order in ORDERS for visit in sessions
                for kind in ("rows", "labels", "secs")}
        if set(rows.files) != keys:
            raise ValueError(f"missing or extra orders/visits in {path}")
        for order in ORDERS:
            for session, delivery in harness.plans(data, order, streams).items():
                s = session.split("|")[1]
                decisions = rows[f"labels_{order}_{s}"]
                if (not np.array_equal(rows[f"rows_{order}_{s}"], delivery)
                        or decisions.shape != delivery.shape or decisions.dtype != np.int8
                        or not np.isin(decisions, (0, 1, 2)).all()):
                    raise ValueError(f"invalid canonical rows or decisions in {path}, order {order}, visit {s}")


def gather(shards, out, data_path):
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = harness.load(data_path)
    found = set()
    with TemporaryDirectory(prefix=".cap250-gather-", dir=out.parent) as tmp:
        pending = Path(tmp) / "scores"
        pending.mkdir()
        for shard in shards:
            for path in sorted(shard.glob("pred_*_*.npz")):
                mode, person = path.stem.split("_")[1:]
                key = mode, person
                if key not in FOLDS or key in found:
                    raise ValueError(f"unexpected or duplicate fold: {path}")
                check_prediction(data, path, mode, person)
                shutil.copyfile(path, pending / path.name)
                found.add(key)
        if found != FOLDS:
            raise ValueError(f"folds missing from cluster shards: {sorted(FOLDS - found)}")
        subprocess.run([sys.executable, "-B", str(ROOT / "cleanslate/harness/score.py"),
                        str(pending), "--data", str(data_path)], check=True)
        pending.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data())
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    gather(args.shards, args.out, args.data)
