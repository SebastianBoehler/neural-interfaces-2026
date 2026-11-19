"""Pinned paths and source-only calibration rows for returning Eye."""

import numpy as np

from eye_scope import BASE, DATA, DATA_SHA, FOLDS, ORDER_FILE, ORDER_SHA, pin


FIT_SOURCE = "5cd02b4aa213c6ce72c167cd1ddd8408447793a8cab54c3d998ecf9ba7d46343"
CAL = BASE / "ni26_friday_return_eye_cal_head_v1_20261119"
TARGET = BASE / "ni26_friday_return_eye_cal_native_v1_20261119"
MASS = .5
RIDGE_C = .10
DIMENSION = 384


def own_visit(who):
    pin(DATA, DATA_SHA)
    pin(ORDER_FILE, ORDER_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        people, days = z["subject"], z["session"]
    with np.load(ORDER_FILE, allow_pickle=False) as z:
        order = z["full|0|all_loso"]
    own = (people == f"Dreyer2026Proteus/{who}") & (days == "1")
    rows = order[own[order]]
    if len(rows) != FOLDS[who][3] or np.array_equal(rows, np.flatnonzero(own)):
        raise ValueError("no genuine source-only visit-1 order")
    np.testing.assert_array_equal(np.sort(rows), np.flatnonzero(own))
    return rows
