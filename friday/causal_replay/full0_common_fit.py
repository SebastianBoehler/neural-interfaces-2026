"""Run the genuine held-new common-source full0 refits."""

from functools import partial

from full0_common_case import BRIDGE, BRIDGE_SOURCE, ORDER
from within0_matched_fit import main
from within0_pair_inputs import authenticate


if __name__ == "__main__":
    main(ORDER, partial(authenticate, order=ORDER, bridge=BRIDGE, source_sha=BRIDGE_SOURCE))
