"""Fit switching gain on actual overlap decisions, using source rows only."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / name) for name in ("runner_gate_wave", "fbc_protect_wave")]
from headroom_margin import row_weights
from runner_model import MIN_GAIN, _features, _fit_rows, disagreement

ARMS = ("direct", "confidence")
MIN_FIT_ROWS = 30


def describe(trace, arm):
    final, votes, reps = (trace[k] for k in ("final", "votes", "reps"))
    winner, runner, broad, margin = disagreement(final, votes)
    row = np.arange(len(final))
    incumbent = np.column_stack((np.ones(len(final)), margin, final[row, runner],
                                 (votes == runner[:, None]).mean(1),
                                 (votes == winner[:, None]).mean(1)))
    x = np.column_stack((incumbent, _features(reps, winner, runner, arm)))
    return x, winner, runner, broad


def fit(trace, meta, labels, group):
    """Accept only matching source labels, then fit on broad disagreement rows."""
    rows = trace["indices"]
    if labels.shape != (len(rows),) or group.shape != labels.shape:
        raise ValueError("Expected source labels and groups aligned to the replay")
    w = row_weights(labels, *(meta[k][rows] for k in ("subject", "session", "interface")))
    fitted = {}
    for arm in ARMS:
        x, winner, runner, broad = describe(trace, arm)
        y = (labels == runner).astype(float) - (labels == winner)
        masks = [broad, *(broad & (group != g) for g in range(3))]
        if any(m.sum() < MIN_FIT_ROWS or set(y[m]) != {-1, 0, 1} for m in masks):
            raise ValueError("Actual source disagreements lack fit support or outcome coverage")
        coefs = np.stack([_fit_rows(x[m], y[m], w[m]) for m in masks])
        fitted[arm] = {"coef": coefs.tolist(), "fit_rows": [int(m.sum()) for m in masks],
                       "outcomes": {str(t): int((broad & (y == t)).sum()) for t in (-1, 0, 1)}}
    return fitted


def decisions(trace, fitted):
    labels = {}
    for arm in ARMS:
        x, winner, runner, broad = describe(trace, arm)
        predictions = np.einsum("nd,kd->nk", x, np.asarray(fitted[arm]["coef"])).clip(-1, 1)
        labels[arm] = np.where(broad & (predictions.min(1) >= MIN_GAIN), runner, winner)
        if arm == "direct":
            labels["direct_point"] = np.where(broad & (predictions[:, 0] >= MIN_GAIN), runner, winner)
    labels["identity"] = winner
    return labels, broad, runner
