"""Person-disjoint partitioning and pinned causal session ordering."""
import numpy as np


ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")


def partition(people, held, seed):
    people = np.asarray(people)
    available = sorted(set(people.tolist()) - {held})
    if held not in set(people.tolist()) or len(available) != 9 or seed not in (0, 1):
        raise ValueError("expected one of ten people and seed 0 or 1")
    val_person = available[seed % len(available)]
    test = np.flatnonzero(people == held)
    val = np.flatnonzero(people == val_person)
    train = np.flatnonzero((people != held) & (people != val_person))
    assert len(set(people[train])) == 8
    return train, val, test


def ranks(order, n):
    order = np.asarray(order)
    if order.shape != (n,) or not np.issubdtype(order.dtype, np.integer) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("order must be a full row permutation")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


def ordered_streams(rows, stream, rank):
    rows = np.asarray(rows)
    stream, rank = np.asarray(stream), np.asarray(rank)
    if len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= len(rank)).any():
        raise ValueError("rows must be unique and in range")
    return [r[np.argsort(rank[r], kind="stable")] for s in dict.fromkeys(stream[rows])
            if len(r := rows[stream[rows] == s])]


def aligned_rows(model, rows, rank, *, check_prefix=True):
    """Return aligned data in input row order, resetting alignment at each session."""
    rows = np.asarray(rows)
    result = np.empty((len(rows), *model.d["X"].shape[1:]), dtype=np.float32)
    positions = np.full(len(rank), -1, dtype=np.int64)
    positions[rows] = np.arange(len(rows))
    for r in ordered_streams(rows, model.d["stream"], rank):
        value = model._aligned(r, causal=True)
        if check_prefix and len(r) > 1:
            n = len(r) // 2
            if not np.allclose(value[:n], model._aligned(r[:n], causal=True), atol=1e-5, rtol=1e-5):
                raise ValueError("future window changed an earlier normalized input")
        result[positions[r]] = value
    if not np.isfinite(result).all():
        raise ValueError("non-finite aligned input")
    return result
