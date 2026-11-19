"""Give the canonical NI-V Scherer export a single scoring interface per visit."""
import argparse
from pathlib import Path

import numpy as np

from data import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    if sha(a.source) != "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91":
        raise ValueError("Expected the audited Scherer HAND/SUB/WORD export")
    with np.load(a.source, allow_pickle=False) as f:
        out = {k: f[k] for k in ("X", "y", "subject", "session", "ch_names")}
    out["interface"] = np.repeat("Scherer", len(out["y"]))
    np.savez(a.destination, **out)


if __name__ == "__main__":
    main()
