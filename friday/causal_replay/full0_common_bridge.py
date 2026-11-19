"""Rebuild the two actual full0 shadows and strict-past refit targets."""

from full0_common_case import FBL, FBL_SOURCE, ORDER, ROUTE, ROUTE_SHA, ZERO, ZERO_SHA, ZERO_SOURCE, qualified_zero
from within0_sinc_bridge import run


if __name__ == "__main__":
    qualified_zero(ZERO_SHA, ORDER, ZERO, ZERO_SOURCE)
    run(ORDER, ZERO, ZERO_SHA, ROUTE, ROUTE_SHA, FBL, FBL_SOURCE, prior_is_terminal=True)
