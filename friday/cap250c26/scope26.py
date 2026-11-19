"""Pin two fresh native Scherer deliveries."""

import numpy as np

from data import permutation
from settings import stamp

SEEDS=(18411,19411)
HASHES=("2cb1725d8e5ea4cd3de697e880c260135348ae083b816ea278446d83ffc954ab",
        "94f721ebb65eb2c8fa4868de948fd66c818c92fa6ca69fd405611b694fdfca53")
ARMS=("stock","peer","sham")


def schedule(subject,visit):
    group=np.char.add(np.char.add(subject,"|"),visit)
    assert group.shape==(2130,) and len(set(group))==18 and len(set(subject))==9
    out={}
    for seed,digest in zip(SEEDS,HASHES,strict=True):
        rows=permutation(group,None,"full",seed).astype("<i8")
        assert stamp(rows)==digest
        out[seed]={k:rows[group[rows]==k] for k in sorted(set(group))}
    return out
