"""Lane G array indices -> tasks. python jobs.py ZIP OUT KIND IDX   (KIND: feats 0-3, train 0-15, train2 0-11, prod 0-9)
feats: one task per held visit (the 8 pinned orders in sequence); train: person x training set x arm; prod: person (g1)."""
import sys

import tasks as T
import common as K

MODES = ("stock", "g1", "g1c", "g1a")  # indices 0-15 are frozen: the first release's array maps to them
MODES2 = ("stock", "g2", "g4")  # amendment 1: train2 0-11
FEATS = [(p, v) for p in ("P04", "P10") for v in (2, 3)]
TRAIN = [(p, vs, m) for p in ("P04", "P10") for vs in ("1", "12") for m in MODES]
TRAIN2 = [(p, vs, m) for p in ("P04", "P10") for vs in ("1", "12") for m in MODES2]
PROD = ["P%02d" % i for i in range(1, 11)]


def main(zip_path, out, kind, idx):
    idx = int(idx)
    if kind == "feats":
        for order in K.ORDERS:
            T.cmd_feats(zip_path, out, FEATS[idx][0], FEATS[idx][1], order)
    elif kind == "train":
        T.cmd_train(zip_path, out, *TRAIN[idx])
    elif kind == "train2":
        T.cmd_train(zip_path, out, *TRAIN2[idx])
    elif kind == "prod4":
        T.cmd_train(zip_path, out, PROD[idx], "", "g4", prod=True)
    elif kind == "prod":
        T.cmd_train(zip_path, out, PROD[idx], "", "g1", prod=True)
    else:
        raise SystemExit(f"unknown kind {kind}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
