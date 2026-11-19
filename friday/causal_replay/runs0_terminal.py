"""Predict, seal and separately score the two complete runs0 systems."""

import argparse
from functools import partial
from pathlib import Path

from bridge_owner import sha
from runs0_case import BRIDGE, BRIDGE_SOURCE, ORDER, PINS, ZERO, ZERO_SOURCE
from within0_complete import predict, score
from within0_pair_inputs import authenticate


verify = partial(authenticate, order=ORDER, bridge=BRIDGE, source_sha=BRIDGE_SOURCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("score")
    live = modes.add_parser("predict")
    live.add_argument("matrix", type=Path)
    live.add_argument("matrix_sha")
    args = parser.parse_args()
    if args.mode == "predict":
        predict(sha(ZERO), args.matrix, args.matrix_sha, sha(BRIDGE),
                order=ORDER, pins=PINS, zero=ZERO, zero_source=ZERO_SOURCE, rest_seeds=3, verify=verify)
    else:
        score(order=ORDER, zero=ZERO, zero_source=ZERO_SOURCE, bridge=BRIDGE, verify=verify)
