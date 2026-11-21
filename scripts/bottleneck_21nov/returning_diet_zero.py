"""Verify the production diet and em_half on two held-new folds."""
import argparse
from functools import partial
from pathlib import Path

from returning_diet import ARMS, PARENT_SHA, patch
from returning_em import patch as patch_em
from returning_em_zero import check_zero, verify_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "candidate", "run", "data", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("sha")
    a = parser.parse_args()
    change = partial(patch, weight=ARMS[a.arm])

    def fold_sources(source):
        parent = patch_em(source, 0.5)
        return parent, change(parent)

    verify = partial(verify_pair, a.parent, a.candidate, expected_sha=PARENT_SHA, patch_source=change)
    check_zero(a, verify, fold_sources)


if __name__ == "__main__":
    main()
