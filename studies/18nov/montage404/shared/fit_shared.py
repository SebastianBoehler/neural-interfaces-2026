"""Shared person-specific task contrast directions, with matched donor controls."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
PARENT = ROOT / 'ni26_align404_v1_20261118'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit


def directions(graz, brain, shift):
    a = graz - graz.mean(0)
    b = np.roll(brain - brain.mean(0), shift, axis=0)
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    cross = a.T @ b / len(a)
    vals, vecs = np.linalg.eigh((cross + cross.T)/2)
    fit.check(vals[-3] > 1e-8, 'positive shared directions')
    return vecs[:, -3:], vals[-3:].tolist()


def agreement(graz, brain, j):
    keep = np.arange(len(graz)) != j
    a = graz[j] - graz[keep].mean(0)
    b = brain[j] - brain[keep].mean(0)
    vals = []
    for shift in range(5):
        basis, _ = directions(graz[keep], brain[keep], shift)
        x, y = a @ basis, b @ basis
        vals.append(np.sum(x*y) / (np.linalg.norm(x)*np.linalg.norm(y)))
    return vals[0] - np.mean(vals[1:])


def main(smoke):
    fit.check(fit.digest(PARENT/'source.sha256') == '23762a6bbea74582a1553bdbc26f28c1cdcf21a4b5c07b9ac8ba1ce429f92309', 'previous source')
    for line in (PARENT/'source.sha256').read_text().splitlines():
        h, name = line.split(maxsplit=1)
        fit.check(fit.digest(PARENT/name) == h, 'parent checksum')
    # Supply the two changes at the owning operations of the identical source fit.
    fit.run(smoke, directions, agreement, amplify=True)
    if not smoke:
        with np.load('members.npz') as z, np.load(PARENT/'members.npz') as baseline:
            fit.check(np.array_equal(z['index'], baseline['index']) and
                      np.array_equal(z['O'], baseline['O']), 'ordinary model parity')


if __name__ == '__main__':
    main(sys.argv[1] == 'smoke')
