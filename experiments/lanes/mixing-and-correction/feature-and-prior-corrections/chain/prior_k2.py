"""Lane K2 idea 5: feasible class prior from past votes (registered in evidence/wave_20nov/laneK2_ideas456.md).

Source statistics come from fit-excluded pre-rebalance mix posteriors of other people (labels only here). Per session, from votes strictly before window i,
every EVERY calls after NMIN windows: feasible set {pi on the 0.01 simplex grid: |q_c - (C pi)_c| <= eps_n for all categories of hard and high-margin votes}."""
import numpy as np

GRID = np.array([(i / 100, j / 100, (100 - i - j) / 100) for i in range(101) for j in range(101 - i)])
U = np.full(3, 1 / 3)
NMIN, EVERY, NARROW = 40, 20, 0.20
UNIFORM, INFEASIBLE, WIDE, ACTIVE = 0, 1, 2, 3  # status codes


def votes(mix, t):
    srt = np.sort(mix, 1); hard = mix.argmax(1)
    return hard, np.where(srt[:, -1] - srt[:, -2] >= t, hard, 3)


def source_stats(mix, y, mask):
    """hard-vote confusion Ch[j,k] = P(vote j | true k) (3x3), high-margin vote confusion Cm (4x3, category 3 = no confident vote), margin threshold = source median."""
    srt = np.sort(mix[mask], 1); t = float(np.median(srt[:, -1] - srt[:, -2]))
    hard, hm = votes(mix, t)
    Ch, Cm = np.zeros((3, 3)), np.zeros((4, 3))
    for k in range(3):
        r = mask & (y == k)
        Ch[:, k] = np.bincount(hard[r], minlength=3) / r.sum(); Cm[:, k] = np.bincount(hm[r], minlength=4) / r.sum()
    return dict(Ch=Ch, Cm=Cm, t=t)


def feasible(hard, hm, n, st, delta):
    eps = np.sqrt(np.log(2 * 7 / delta) / (2 * n))
    qh, qm = np.bincount(hard, minlength=3) / n, np.bincount(hm, minlength=4) / n
    ok = (np.abs(GRID @ st["Ch"].T - qh).max(1) <= eps) & (np.abs(GRID @ st["Cm"].T - qm).max(1) <= eps)
    return GRID[ok]


def session_prior(mix_s, st, delta):
    """per window of one session (delivery order): the prior pi_i in force at window i (from votes before i) and its status code."""
    n = len(mix_s); hard, hm = votes(mix_s, st["t"])
    pi, status = np.tile(U, (n, 1)), np.zeros(n, dtype=int); cur, code = U, UNIFORM
    for i in range(n):
        if i >= NMIN and i % EVERY == 0:
            F = feasible(hard[:i], hm[:i], i, st, delta)
            if len(F) == 0:
                cur, code = U, INFEASIBLE
            elif (F.max(0) - F.min(0)).min() >= NARROW:
                cur, code = U, WIDE
            else:
                cur, code = F[np.argmin(((F - U) ** 2).sum(1))], ACTIVE
        pi[i], status[i] = cur, code
    return pi, status


def responsibilities(mix_s, pi):
    r = mix_s * pi / U
    return r / r.sum(1, keepdims=True)


def diagnose(rows):
    """rows: list of (age i, status, L1 error of the prior in force, L1 error of uniform) -> summary by prefix age."""
    a = np.asarray(rows, dtype=float); out = {}
    for name, lo, hi in (("40-80", 40, 80), ("80-120", 80, 120), ("120+", 120, 10 ** 9)):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.any():
            out[name] = dict(n=int(m.sum()), active=float((a[m, 1] == ACTIVE).mean()), infeasible=float((a[m, 1] == INFEASIBLE).mean()),
                             wide=float((a[m, 1] == WIDE).mean()), l1_prior=float(a[m, 2].mean()), l1_uniform=float(a[m, 3].mean()))
    out["all_windows"] = dict(n=len(a), active=float((a[:, 1] == ACTIVE).mean()))
    return out
