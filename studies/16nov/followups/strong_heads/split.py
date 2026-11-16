"""Deterministic person-disjoint nested selection splits."""

import numpy as np


def source_partition(people, outer_person, seed):
    people = np.asarray(people)
    unique = np.unique(people)
    if len(unique) != 10 or outer_person not in unique or seed not in (0, 1):
        raise ValueError('expected ten people, a valid outer person and seed 0 or 1')
    source = unique[unique != outer_person]
    inner_person = source[seed % len(source)]
    outer = people == outer_person
    inner = people == inner_person
    train = ~(outer | inner)
    if len(np.unique(people[train])) != 8 or np.any(inner & outer):
        raise ValueError('invalid nested person split')
    return train, inner, outer
