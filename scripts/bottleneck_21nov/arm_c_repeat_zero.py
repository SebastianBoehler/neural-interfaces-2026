"""Measure two independent em_half runs through the existing Turing zero check."""
import argparse
from pathlib import Path
from zipfile import ZipFile

from returning_diet import PARENT_SHA
from returning_em import patch
from returning_em_zero import check_zero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "run", "data", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.sha = PARENT_SHA

    def verify_archive():
        with ZipFile(args.candidate) as archive:
            invalid = archive.testzip()
        if invalid is not None:
            raise ValueError(f"invalid em_half member: {invalid}")

    def sources(source):
        half = patch(source, 0.5)
        return half, half

    check_zero(args, verify_archive, sources)


if __name__ == "__main__":
    main()
