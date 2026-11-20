"""Fit and authenticate the C1 encoder for an outer or nested holdout."""
import argparse
from pathlib import Path
import subprocess
import sys
from itertools import combinations

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent / "harness"), str(HERE.parent / "teamC")]
import data as D
import submission as C

HOLDOUTS = [(p,) for p in D.PEOPLE] + list(combinations(D.PEOPLE, 2))
MEMBERS = ",".join(f"{arch}:{seed}" for arch in C.ARCHS for seed in (0, 1))


def fit_path(root, held):
    return root / "base" / "_".join(sorted(held))


def roster(held):
    return [p for p in D.PEOPLE if p not in held]


def load_base(root, held, epochs, device, **cfg):
    path = fit_path(root, held)
    blob = torch.load(path / "weights.pt", map_location="cpu", weights_only=False)
    members = ",".join(f"{m['arch']}:{m['seed']}" for m in blob["members"])
    args = blob["args"]
    if (blob["people"] != roster(held) or members != MEMBERS or args["epochs"] != epochs
            or args["extra"] or args["exclude"] or args["rot"] != 0):
        raise ValueError(f"C1 encoder provenance mismatch at {path}")
    return C.load_package(path, device, cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("task", type=int)
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M1 must run on a Slurm GPU node")
    held = HOLDOUTS[args.task]
    out = fit_path(args.root, held)
    subprocess.run([
        sys.executable, "-u", str(HERE.parent / "teamC" / "train.py"),
        "--people", ",".join(roster(held)), "--out", str(out),
        "--members", MEMBERS, "--epochs", str(args.epochs), "--data", D.DATA,
    ], check=True)
    load_base(args.root, held, args.epochs, "cuda:0")
    print("C1 roster and weights verified", held, flush=True)


if __name__ == "__main__":
    main()
