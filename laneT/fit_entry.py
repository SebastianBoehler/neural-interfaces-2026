"""usage: fit_entry.py ORDER ARM FAMILY BRIDGE_SHA [--smoke]   (run from an arm directory = a copy of the pinned matched-fit
release; ARM ctl = the unmodified stock stream, the zero-change control). Replaces only run_stream."""
import sys
from functools import partial

import within0_matched_fit as W
from ttt_core import make_run_stream

order, arm = sys.argv[1], sys.argv[2]
del sys.argv[1:3]
if arm != "ctl":
    W.run_stream = make_run_stream(arm)
if order == "full0":
    from full0_common_case import BRIDGE, BRIDGE_SOURCE, ORDER
    from within0_pair_inputs import authenticate
    W.main(ORDER, partial(authenticate, order=ORDER, bridge=BRIDGE, source_sha=BRIDGE_SOURCE))
else:  # the within0 release has no main(): same command line, run(family, bridge_sha, smoke)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=W.FAMILIES)
    ap.add_argument("bridge_sha")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    W.run(a.family, a.bridge_sha, a.smoke)
