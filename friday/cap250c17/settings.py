"""Fixed experiment arms and independently anchored Scherer session deliveries."""

import hashlib

import numpy as np

from data import permutation


ARMS = ("stock", "permuted", "informative")
ACTIVE = ARMS[1:]
SEEDS = (6411, 7411)
ORDERS = {
    411: "fc5fb76599716457329327e80e9982bcc02360860c824f692f8880ec72e9814a",
    1411: "a4e996f5aa150227875aae7e624d7bb29c6618ecfcda9cffedbfff480ddacdbe",
    6411: "774cd0e2d1a7c3772042978b6533a3b86d94b5540cf65d2c9c87a1693e73389b",
    7411: "ecabf14fc4028df52bac13b18ee75eb83e79be6456ea94b41e17d6c20229906d",
}


def schedule(people, days):
    s = np.char.add(np.char.add(people,"|"),days)
    assert s.shape == (2130,) and len(set(people)) == 9 and len(set(s)) == 18
    out = {}
    for key, want in ORDERS.items():
        index = permutation(s,None,"full",key).astype("<i8")
        assert hashlib.sha256(index.tobytes()).hexdigest() == want
        out[key] = {visit: index[s == visit] for visit in sorted(set(s))}
    return out
