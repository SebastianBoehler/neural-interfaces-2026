"""Warm paired member timings and numerical checks. Run via check_cb_speed.py."""

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch

ATOL = 1e-5


def check_graph(cb, expected):
    state = cb._forward_graph
    if state.graph is None or state.failed or state.replays != expected:
        raise RuntimeError(f"expected {expected} graph replays, got {state.replays}, failed={state.failed}")


def stats(times):
    return dict(n=len(times), mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times, 95)))


def timed(fn, data):
    start = perf_counter()
    output = fn(data)
    torch.cuda.synchronize()
    return output, 1e3 * (perf_counter() - start)


def benchmark(models, x, sessions, smoke=False):
    old, new = models
    eager = deepcopy(new.cb)
    arms = dict(original=old.cb, eager=eager, graph=new.cb)
    delta = dict(eager=0.0, graph=0.0)
    times = {key: [] for key in (*arms, "hard")}
    cold = {key: [] for key in times}
    for cb in arms.values():
        cb.reset()
    eager._forward_graph.failed = True
    warm = next(iter(sessions.values()))[:(2 if smoke else 100)]
    for i in warm:
        for cb in arms.values():
            cb.prob(x[i])
        new.hard_probability(x[i])
    for group, rows in enumerate(sessions.values()):
        if smoke and group:
            break
        if smoke:
            rows = rows[:8]
        for cb in arms.values():
            cb.reset()
        eager._forward_graph.failed = True
        new.hard_actors = None
        sequence = tuple(arms) if group % 2 == 0 else tuple(reversed(arms))
        for age, i in enumerate(rows):
            row = x[i]
            got = {}
            for key in sequence:
                got[key], elapsed = timed(arms[key].prob, row)
                (times if age else cold)[key].append(elapsed)
            _, elapsed = timed(new.hard_probability, row)
            (times if age else cold)["hard"].append(elapsed)
            for key in delta:
                delta[key] = max(delta[key], float(np.max(np.abs(got[key] - got["original"]))))
        check_graph(new.cb, len(rows))

    if max(delta.values()) > ATOL:
        raise RuntimeError(f"CBraMod parity exceeds {ATOL}: {delta}")
    return dict(warm={key: stats(values) for key, values in times.items()},
                first={key: stats(values) for key, values in cold.items()}, max_abs=delta,
                windows=len(times["graph"]) + len(cold["graph"]), sessions_count=len(cold["graph"]))
