"""Label-free output changes shared by the review scorer and package patch."""

import numpy as np

COUNT_DOSES = (0.0, 0.5, 1.0)
BALANCE_DOSES = (0.0, 0.35, 0.70, 1.0)


def shrink_count(raw, counted, recognised, strength):
    if not recognised or strength == 0.0:
        return counted
    if strength == 1.0:
        return raw
    return (1 - strength) * counted + strength * raw


def balance_output(raw, history, recognised, strength):
    if recognised or strength == 0.0:
        return raw
    history.append(np.asarray(raw, dtype=np.float64))
    if len(history) < 20:
        return raw
    p = np.maximum(history, 1e-12)
    scale = np.ones(3)
    for _ in range(12):
        q = p * scale
        q /= q.sum(axis=1, keepdims=True)
        mass = q.mean(axis=0)
        if np.max(np.abs(mass - 1 / 3)) <= 1e-5:
            break
        scale /= 3 * mass
        scale /= scale.max()
    log_scale = np.log(scale)
    return raw * np.exp(strength * np.clip(log_scale - log_scale.mean(), -0.35, 0.35))
