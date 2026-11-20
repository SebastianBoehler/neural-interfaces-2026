"""Fit the cap250 SessionBN family sbn_fbc_ea (fb_sbn_rot05) and export its stateful traces."""

from _sbn import run

if __name__ == "__main__":
    run(__doc__, "fb_sbn_rot05", ((35, "sbn_k2", 2), (41, "sbn_k5", 5)))
