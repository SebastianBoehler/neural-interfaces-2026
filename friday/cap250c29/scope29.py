"""Two locked full-session Scherer orders."""
import numpy as np

from data import permutation
from files29 import stamp

SEEDS=(24411,25411)
HASHES=("28ef69fb365211f969c5e7815778ef00eafbb774de22e83513323fe1216cbf8e",
        "4fda84f365e64c3a2053720bed950413ec1a1eece014f3442fdfddc4fe43dff3")


def schedule(person,visit):
    group=np.char.add(np.char.add(person,"|"),visit)
    assert group.shape==(2130,) and len(set(person))==9 and len(set(group))==18
    result={}
    for seed,want in zip(SEEDS,HASHES,strict=True):
        p=permutation(group,None,"full",seed).astype("<i8")
        assert stamp(p)==want
        result[seed]={k:p[group[p]==k] for k in sorted(set(group))}
    return result
