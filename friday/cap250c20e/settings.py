"""Fixed native worker scope and authenticated Scherer orders."""

import hashlib
from pathlib import Path

import numpy as np

from data import permutation

IDS = tuple(range(47, 53))
SEEDS = (8411, 9411)
ORDERS = {
    0: "63ef8f828ff49cfa2b2a9152571b5a496810d1d7834b2f38e2be1c345307c5d0",
    8411: "e62051f2e4138f1255563fdfecb77dc0420489154d191e1bdd7bd791e10282ec",
    9411: "e58832af5b78c1ac30b5e08ae7a967e09551d693961fe0e519d269903223dfbb",
}
ACTORS = ("0dc677091668ad8acc6716a9bd75df6371829aa13ac6cd90985a3ef675126cdd",
          "cb16ec5919c8fc58d7a9a9f4c8defcb425a16b4d36359687a0a68a1b6b078634")
UPPER = 5e-5
ARMS = ("stock", "hard")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stamp(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def schedule(people, visits):
    stream = np.char.add(np.char.add(people,"|"),visits)
    assert stream.shape==(2130,) and len(set(people))==9 and len(set(stream))==18
    by_seed={}
    for seed, expected in ORDERS.items():
        ix=permutation(stream,None,"full",seed).astype("<i8")
        assert stamp(ix)==expected
        by_seed[seed]={s:ix[stream[ix]==s] for s in sorted(set(stream))}
    return by_seed
