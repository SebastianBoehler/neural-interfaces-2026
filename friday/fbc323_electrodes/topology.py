"""Authenticate a directed electrode graph and its degree-matched scrambled control."""
from pathlib import Path

import numpy as np

from anchors import row_sha, sha


HERE = Path(__file__).resolve().parent
MONTAGE = "71d07f1263c348b64990c2d1c329cd30fc21281583563be5a8629ed9fcf281a9"
SENSORS, DEGREE = 41, 4
SLOTS = np.arange(SENSORS)[:, None]


def topology(channels, independent):
    release = "ni26_friday_s1c_sensor_20261118" if independent else "friday_s1c_sensor_20261118"
    path = HERE.parent / release / "montage.tsv"
    if sha(path) != MONTAGE:
        raise ValueError("the previously authenticated F18 montage differs")
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:]]
    if [r[0] for r in rows] != list(channels) or len(rows) != SENSORS:
        raise ValueError("electrodes do not match official montage order")
    xyz = np.array([[float(v) for v in r[1:]] for r in rows])
    distance = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    physical = np.argsort(distance, axis=1, kind="stable")[:, 1:DEGREE+1]
    scrambled = physical.copy()
    choices = np.random.default_rng(27027).integers((0,0,0,0),(SENSORS,DEGREE,SENSORS,DEGREE),size=(10000,4))
    for a, a_slot, c, c_slot in choices:
        b, d = scrambled[a, a_slot], scrambled[c, c_slot]
        if a != c and a != d and c != b and d not in scrambled[a] and b not in scrambled[c]:
            scrambled[a, a_slot], scrambled[c, c_slot] = d, b
    edges_changed = float((~(scrambled[:, :, None] == physical[:, None, :]).any(-1)).mean())
    lengths = [float(np.median(distance[SLOTS, g])) for g in (physical, scrambled)]
    degree = [np.bincount(g.ravel(), minlength=SENSORS) for g in (physical, scrambled)]
    if (any((g == SLOTS).any() or any(len(set(r)) != DEGREE for r in g) for g in (physical, scrambled)) or
            not np.array_equal(*degree) or edges_changed <= .80 or lengths[1] < 1.5 * lengths[0]):
        raise ValueError("real and scrambled topology degree or distance controls differ")
    return dict(hard=None, physical=physical, scrambled=scrambled), dict(
        montage=MONTAGE, edges_changed=edges_changed, median_distances=lengths,
        incoming_degree=row_sha(degree[0]), graph_physical=row_sha(physical), graph_scrambled=row_sha(scrambled))
