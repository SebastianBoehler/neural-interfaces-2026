"""Pooling arithmetic shared by the Fold replay and the ZIP builder."""
import numpy as np


def round3_log_pool(log_a, log_b, w):
    return (1 - w) * log_a + w * log_b


def round3_log_post(p):
    return np.log(np.clip(p, 1e-12, None))


def round3_mix(p, q, weight):
    lp = round3_log_pool(round3_log_post(p), round3_log_post(q), weight)
    v = np.exp(lp - lp.max())
    return v / v.sum()
