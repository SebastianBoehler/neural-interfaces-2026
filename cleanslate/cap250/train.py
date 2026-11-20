"""Train one cap250 fold inside a Slurm GPU allocation; use graph.py for parallel fits."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from assemble import assemble
from calibrate import run as stream_calibration
from components import Variant, part_path
from route_inputs import b2_fits, subfold
from source import add_scope, validate_scope
from worker import fit


def train(people, first, data, out):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("train.py runs inside a Slurm allocation. Use graph.py --submit for the parallel DAG.")
    if out.exists():
        raise FileExistsError(out)
    validate_scope(people, first)
    variant = Variant.from_env()
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cap250-train-", dir=out.parent) as tmp:
        root = Path(tmp)
        banks = {}
        for held in people:
            inner, inner_first = subfold(people, first, held)
            parts = root / "nested" / held / "parts"
            for family, seed in b2_fits():
                fit(family, seed, inner, inner_first, data, part_path(parts, family, seed), base_only=family == "geometry")
            bank = parts.parent / "bank.npz"
            stream_calibration(inner, inner_first, data, parts, held, held in first, bank)
            banks[held] = str(bank)
        manifest = root / "route.json"
        manifest.write_text(json.dumps(banks))
        os.environ["CAP250_ROUTE_OOF"] = str(manifest)
        parts = root / "parts"
        for family, seed in variant.fits():
            fit(family, seed, people, first, data, part_path(parts, family, seed))
        assemble(people, first, data, parts, out, variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_scope(parser)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train(args.people, args.first_session_only, args.data, args.out)
