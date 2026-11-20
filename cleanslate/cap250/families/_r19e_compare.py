"""Compare fresh R19e family fits with the archived v5.1b member arrays (max abs diff per net, decision-relevant)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import ORDINARY, FAMILIES  # noqa: E402


def main(parts, archive):
    ref = np.load(archive, allow_pickle=False)
    for family, base in ORDINARY.items():
        for seed in range(FAMILIES[family].seeds):
            f = Path(parts) / family / f"s{seed}" / "weights.npz"
            if not f.exists():
                print(family, seed, "MISSING"); continue
            new = np.load(f, allow_pickle=False)
            pre = f"net{base + seed}_"
            keys = sorted(k for k in ref.files if k.startswith(pre))
            if sorted(new.files) != keys:
                print(family, seed, "KEYSET DIFFERS", sorted(set(new.files) ^ set(keys))[:6]); continue
            d = {k: float(np.abs(new[k].astype(np.float64) - ref[k].astype(np.float64)).max()) for k in keys
                 if new[k].dtype.kind in "fiub"}
            same = all(np.array_equal(new[k], ref[k]) for k in keys)
            print(family, seed, "BITWISE" if same else f"max|d|={max(d.values()):.3e}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
