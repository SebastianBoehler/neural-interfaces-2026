"""Seal the shared FBLight and Sinc matched runs0 targets in a single process."""

from runs0_case import BRIDGE_OLD, BRIDGE_OLD_SHA, FBL, FBL_SOURCE, ORDER, ROUTE, ROUTE_SHA
from within0_sinc_bridge import run


if __name__ == "__main__":
    run(ORDER, BRIDGE_OLD, BRIDGE_OLD_SHA, ROUTE, ROUTE_SHA, FBL, FBL_SOURCE)
