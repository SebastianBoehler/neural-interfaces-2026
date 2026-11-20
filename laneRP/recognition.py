"""One-template-per-person open-set gallery. A missing identity is a valid outcome."""
import numpy as np


def gallery(rows):
    return [np.stack([d[k][d["day"] == 1].mean(0, dtype=np.float64) for d in rows])
            for k in ("spec", "conn")]


def distance_ratio(probe, templates):
    # Raw-space Euclidean distance within each view. Gallery-relative ratios are unitless.
    prefix = np.cumsum(probe, axis=0, dtype=np.float64) / np.arange(1, len(probe) + 1)[:, None]
    dist2 = (prefix**2).sum(1)[:, None] + (templates**2).sum(1) - 2 * prefix @ templates.T
    dist = np.sqrt(np.maximum(dist2, 0))
    near = np.argmin(dist, axis=1)
    runner = np.partition(dist, 1, axis=1)[:, 1]
    return near, dist / np.maximum(runner[:, None], 1e-10)


def recognize(probes, templates):
    (a, ra), (b, rb) = [distance_ratio(x, g) for x, g in zip(probes, templates, strict=True)]
    joint = (ra + rb) / 2
    j = joint.argmin(1)
    best = joint[np.arange(len(j)), j]
    result = {}
    for name, match in (("strict", (a == b) & (best < .80)), ("loose", best < .90)):
        valid = np.flatnonzero(match & (np.arange(len(j)) >= 7))
        result[name] = (int(valid[0]), int(j[valid[0]])) if len(valid) else (-1, -1)
    return result
