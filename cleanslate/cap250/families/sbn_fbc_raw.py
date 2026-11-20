"""Fit the cap250 SessionBN family sbn_fbc_raw (fbcnet_raw_sbn) and export its stateful traces."""

from _sbn import run

if __name__ == "__main__":
    run(__doc__, "fbcnet_raw_sbn", ((47, "sbn_fbcnet_raw_k20", 20),))
