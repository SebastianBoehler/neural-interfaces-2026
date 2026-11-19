"""Pin the two new Scherer delivery permutations independently of outcomes."""

import numpy as np

from data import permutation
from readout import stamp


SEEDS=(12411,13411)
DELIVERIES=("15690f869b297ec662c7809773afc8672f4628da88944eb341cee81a99acc05f",
            "feef4f86d00ee1e344ac4a2cf7d760e1bf87340d8764de523109927865dd7f39")


def schedule(subject,visit):
    group=np.char.add(np.char.add(subject,"|"),visit)
    assert group.shape==(2130,) and len(set(group))==18 and len(set(subject))==9
    plans={}
    for seed,digest in zip(SEEDS,DELIVERIES,strict=True):
        p=permutation(group,None,"full",seed).astype("<i8")
        assert stamp(p)==digest
        plans[seed]={k:p[group[p]==k] for k in sorted(set(group))}
    return plans
