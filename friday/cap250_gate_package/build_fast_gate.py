"""Embed the measured covariance-sharing and retirement implementation."""
import argparse
from pathlib import Path

from build_gate import MEASURED, assemble, measured_fragments, revised_with

PARENTS = (
    ("cap250c21_gate/native/live_gate.py", "6af1cb7229dae99834e15a1564b931d9f8610503722a510b10d245119dea3b12",
     ("LiveSelector",), "dcdbd705"),
    ("cap250c21_gcv/shared_cov.py", "24016a105a5efbdbd424738c2cf46b3126d6538f96fa42af1844d21e75e9fca9",
     ("SharedCovSelector",), "da7fe3fc"),
    ("cap250c21_gcv/combined/combined_selector.py", "07c00bdee4a57834fab1b490a9888ec2b5d7eff9a73f5a2b4129ab09d558aaee",
     ("SharedCovRetiringSelector",), "f69664d1"),
)


def revised(raw):
    imports = "from copy import deepcopy\nfrom types import SimpleNamespace"
    common = [part for parent in MEASURED[:2] for part in measured_fragments(*parent)]
    optimized = [part for parent in PARENTS for part in measured_fragments(*parent)]
    return revised_with(raw, "\n\n\n".join([imports, *common, *optimized]), "SharedCovRetiringSelector")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(assemble(args.frozen, args.output, revised))
