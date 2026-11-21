"""Measure the Team B posterior and both fixed-shape feature capture paths."""

from unittest.mock import patch
import warnings

import numpy as np
import torch

from cb_speed_bench import ATOL, stats, timed


def expected_replays(n):
    refresh = [age for age in range(1, n + 1) if age <= 16 or age % 4 == 0]
    return {1: n - len(refresh) + 1, 64: sum(age // 64 for age in refresh)}


def check_features(team, n):
    for size, expected in expected_replays(n).items():
        state = team.feature_graphs[size]
        if (expected and state.graph is None) or state.failed or state.replays != expected:
            raise RuntimeError(f"Team B batch {size} expected {expected} graph replays, got {state.replays}")


def posterior(team, window):
    return team.posterior(torch.from_numpy(window)).cpu().numpy()


def check_fallback(team, windows):
    team.reset()
    expected = np.stack([posterior(team, x) for x in windows])
    check_features(team, len(windows))
    team.reset()
    with warnings.catch_warnings(record=True) as alerts:
        warnings.simplefilter("always", RuntimeWarning)
        with patch.object(torch.cuda, "CUDAGraph", side_effect=RuntimeError("injected Team B capture failure")) as factory:
            got = np.stack([posterior(team, x) for x in windows])
    if factory.call_count != 2 or not all(s.failed and not s.replays for s in team.feature_graphs.values()):
        raise RuntimeError("both Team B capture shapes must use eager after a single failure each")
    if sum("using eager: injected Team B capture failure" in str(a.message) for a in alerts) != 2:
        raise RuntimeError("Team B capture failure was silent")
    np.testing.assert_allclose(got, expected, rtol=0, atol=ATOL)
    team.reset()
    return float(np.max(np.abs(got - expected)))


def benchmark(models, x, sessions, *, graphed, smoke=False):
    arms = [model.team_b for model in models]
    times = [{"incremental": [], "refresh": [], "warm": [], "capture": [], "session": []} for _ in arms]
    worst = 0.0
    warmup = next(iter(sessions.values()))[:(8 if smoke else 100)]
    for team in arms:
        team.reset()
        for i in warmup:
            posterior(team, x[i])
    for group, rows in enumerate(sessions.values()):
        if smoke and group:
            break
        if smoke:
            rows = rows[:72]
        for team in arms:
            team.reset()
        for age, row in enumerate(rows, 1):
            work = "refresh" if age <= 16 or age % 4 == 0 else "incremental"
            got = {}
            for i in ((0, 1) if group % 2 == 0 else (1, 0)):
                got[i], ms = timed(lambda raw: posterior(arms[i], raw), x[row])
                times[i]["session"].append(ms)
                times[i][work].append(ms)
                if age in (1, 64):
                    times[i]["capture"].append(ms)
                else:
                    times[i]["warm"].append(ms)
            if not all(q.shape == (3,) and np.isfinite(q).all() for q in got.values()):
                raise RuntimeError("Team B returned an invalid posterior")
            worst = max(worst, float(np.max(np.abs(got[0] - got[1]))))
        if graphed:
            check_features(arms[1], len(rows))
    if worst > ATOL:
        raise RuntimeError(f"Team B posterior error {worst} exceeds {ATOL}")
    return dict(max_abs=worst, windows=len(times[0]["session"]),
                **{arm: {key: stats(values) for key, values in timings.items()}
                   for arm, timings in zip(("original", "candidate"), times, strict=True)})
