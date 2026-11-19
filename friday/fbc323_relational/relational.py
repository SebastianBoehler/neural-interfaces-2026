"""Define the fixed nested person partitions and relation coordinates."""
from pathlib import Path

import numpy as np
import torch

from anchors import PEOPLE


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "literal", "deranged")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{kind}_{mode}" for kind in KINDS for mode in MODES)
WEIGHT, BETA, DIM, TOL = .25, .10, 1152, 2e-5
F25_ACTOR = "589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c"


def groups(held):
    source = tuple(p for p in PEOPLE if p != held)
    if len(source) != 9:
        raise ValueError("invalid outer held person")
    return tuple(source[i::3] for i in range(3))


def ids(subject):
    person = np.asarray([v.rsplit("/", 1)[-1] for v in subject])
    if sorted(set(person)) != list(PEOPLE):
        raise ValueError("official ten-person roster differs")
    return person


def teacher_rows(person, held, group):
    queried = groups(held)[group]
    query = np.flatnonzero(np.isin(person, queried))
    fit = np.flatnonzero(~np.isin(person, (held, *queried)))
    if len(set(person[fit])) != 6 or set(person[query]) != set(queried) or np.intersect1d(fit, query).size:
        raise ValueError("teacher has invalid admissions or query overlap")
    return fit, query


def tag_for(held, seed, smoke):
    return f"{'smoke_' if smoke else ''}{seed}_{held}"


def unit(z):
    centered = z - z.mean(-1, keepdim=True)
    return torch.nn.functional.normalize(centered, dim=-1)
