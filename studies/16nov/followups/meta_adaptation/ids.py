"""Normalize protocol subject identifiers for the ten fixed LOSO people."""
import numpy as np


def person_ids(subjects):
    subjects = np.asarray(subjects)
    if subjects.ndim != 1: raise ValueError('subject ids must be one-dimensional')
    people = np.asarray([str(value).rsplit('/',1)[-1] for value in subjects])
    expected = {f'P{i:02d}' for i in range(1,11)}
    if set(people) != expected: raise ValueError('unexpected LOSO person identities')
    return people
