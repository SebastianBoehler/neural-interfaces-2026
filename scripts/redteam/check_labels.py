"""Require identical rows and labels for the requested tape's streams."""

import sys

import numpy as np


def check(source, reference):
    with np.load(source, allow_pickle=False) as tape, np.load(reference, allow_pickle=False) as ref:
        streams = [key[5:] for key in tape.files if key.startswith("rows_")]
        if not streams:
            raise ValueError("empty tape")
        total = 0
        for stream in streams:
            for prefix in ("rows", "labels"):
                key = f"{prefix}_{stream}"
                np.testing.assert_array_equal(tape[key], ref[key], err_msg=key)
            recognised = f"recognised_{stream}"
            if recognised in tape and int(tape[recognised]) != -1:
                raise ValueError(f"held-new stream {stream} was recognised")
            total += len(tape[f"rows_{stream}"])
        print(f"PASS exact labels for {total} windows across {len(streams)} streams")


if __name__ == "__main__":
    check(*sys.argv[1:])
