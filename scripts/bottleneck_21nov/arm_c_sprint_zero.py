"""Verify a signed arm c extension on the established 379-window test."""
import argparse
from functools import partial
from pathlib import Path

from arm_c_sprint_policy import ARMS, CANDIDATE_SHA, patch
from diet_decomposition_patch import ARMS as DIET_ARMS, patch as component_patch
from returning_em import patch as patch_half
from returning_em_zero import check_zero, verify_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "candidate", "run", "data", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("sha")
    args = parser.parse_args()
    change = partial(patch, arm=args.arm)

    def fold_sources(source):
        parent = component_patch(patch_half(source, 0.5), DIET_ARMS["c"])
        return parent, change(parent)

    verify = partial(verify_pair, args.parent, args.candidate, expected_sha=CANDIDATE_SHA, patch_source=change)
    check_zero(args, verify, fold_sources)


if __name__ == "__main__":
    main()
