"""Execute and record one cap250 family fit inside a Slurm allocation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from components import FAMILIES, Variant, expected_files
from source import ROOT, add_scope, file_digest, row_digest, scope_args, source_rows


def fit(family, seed, people, first, data, out, *, base_only=False):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cap250 fits must run on a Slurm compute node")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = source_rows(people, first, data)
    smoke = os.environ.get("CAP250_SMOKE") == "1"
    start = time.monotonic()
    with TemporaryDirectory(prefix=f".{family}-s{seed}-", dir=out.parent) as tmp:
        pending = Path(tmp) / "part"
        command = [sys.executable, "-B", str(Path(__file__).parent / "families" / f"{family}.py"),
                   "train", *scope_args(people, first, data), "--seed", str(seed), "--out", str(pending)]
        env = {**os.environ, "CAP250_BASE_ONLY": str(int(base_only))}
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        file_names = set(expected_files(family, seed))
        if base_only and family == "geometry":
            file_names.remove("tpri.npz")
        actual = {path.name for path in pending.iterdir() if path.name != "diagnostics"}
        if actual != file_names:
            raise ValueError(f"unexpected family files: {actual} (expected {file_names})")
        hashes = {name: file_digest(pending / name) for name in sorted(file_names)}
        revision = os.environ.get("CAP250_REV")
        if not revision:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        receipt = dict(family=family, seed=seed, people=people, first_session_only=first,
                       data_sha256=file_digest(data), row_sha256=row_digest(rows), n_rows=len(rows),
                       smoke=smoke, base_only=base_only, variant=Variant.from_env().env(), files=hashes,
                       git_revision=revision, scheduler_job=os.environ["SLURM_JOB_ID"],
                       seconds=round(time.monotonic() - start, 3))
        (pending / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        pending.rename(out)
    print(f"PASS {family} s{seed} {receipt['seconds']:.1f}s {receipt['row_sha256']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-only", action="store_true")
    add_scope(parser)
    args = parser.parse_args()
    if not 0 <= args.seed < FAMILIES[args.family].seeds:
        parser.error(f"invalid seed for {args.family}")
    if args.base_only and args.family != "geometry":
        parser.error("--base-only applies to geometry")
    fit(args.family, args.seed, args.people, args.first_session_only, args.data, args.out, base_only=args.base_only)


if __name__ == "__main__":
    main()
