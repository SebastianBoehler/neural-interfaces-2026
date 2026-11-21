"""Frozen PROTEUS folds and orders for the quarter final check."""

ROSTER = "ba_six"
ORDERS = tuple(range(72000, 72016))
CHUNK = 2
FOLDS = (*(f"new_P{i:02d}" for i in range(1, 11)), "ret_P04", "ret_P10")
CANDIDATE_SHA = "42fb63cf1ae9f54cf48315b00894b3788bc65400e75e06bd208ffe3753c43b07"
