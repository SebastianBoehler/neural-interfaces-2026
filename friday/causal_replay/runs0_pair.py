"""Run a genuine held-new matched runs0 refit using the registered source fits."""

import argparse
from functools import partial

from runs0_case import BRIDGE, BRIDGE_SOURCE, ORDER
from within0_matched_fit import check_smoke, run
from within0_pair_inputs import FAMILIES, authenticate


verify = partial(authenticate, order=ORDER, bridge=BRIDGE, source_sha=BRIDGE_SOURCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("bridge_sha")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-smoke", action="store_true")
    args = parser.parse_args()
    if args.verify_smoke:
        print(check_smoke(args.family, args.bridge_sha, order=ORDER, verify=verify), flush=True)
    else:
        run(args.family, args.bridge_sha, args.smoke, order=ORDER, verify=verify)
