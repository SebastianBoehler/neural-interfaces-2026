"""Run identical replay and require matched ordinary predictions on every row."""
import sys
from pathlib import Path

import numpy as np
import full


def main():
    full.SOURCE = full.ROOT / 'ni26_align404b_v1_20261118'
    full.SOURCE_PIN = '9b4f9cf79557295992a385576e9eea9797e6a394a29753edbc8d70f9bea85db8'
    original = full.ROOT / 'ni26_align404_full_v1_20261118'
    if sys.argv[1] == 'score':
        for j in range(8):
            name = 'order_%d.npz' % j
            with np.load(original / name) as previous, np.load(name) as current:
                for key in ('index', 'O', 'Z'):
                    if not np.array_equal(previous[key], current[key]):
                        raise ValueError(f'unchanged {key} mismatch in order {j}')
    full.main()


if __name__ == '__main__':
    main()
