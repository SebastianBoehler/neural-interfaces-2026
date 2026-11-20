"""Fit the cap250 SessionBN family sbn_fbl (fblight_ea_rot05_sbn) and export its stateful traces."""

from _sbn import run

if __name__ == "__main__":
    run(__doc__, "fblight_ea_rot05_sbn", ((59, "sbn_fblight_ea_rot05_k5", 5),))
