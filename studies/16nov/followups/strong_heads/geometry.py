"""Fold-specific population priors and pinned per-session delivery."""

import numpy as np


def source_prior(priors, counts, excluded):
    priors = np.asarray(priors, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    excluded = tuple(excluded)
    if (priors.shape[0] != 10 or counts.shape != (10,) or np.any(counts <= 0) or
            len(excluded) not in (1, 2) or len(set(excluded)) != len(excluded) or
            any(i not in range(10) for i in excluded) or not np.isfinite(priors).all()):
        raise ValueError('invalid person prior inputs or exclusion set')
    keep = np.array([i for i in range(10) if i not in excluded])
    return np.average(priors[keep], axis=0, weights=counts[keep])


def sessions_in_order(permutation, selected_rows, sessions):
    permutation = np.asarray(permutation, dtype=np.int64)
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    sessions = np.asarray(sessions)
    if (permutation.ndim != 1 or len(sessions) != len(permutation) or
            not np.array_equal(np.sort(permutation), np.arange(len(permutation)))):
        raise ValueError('invalid pinned permutation')
    if (selected_rows.ndim != 1 or not len(selected_rows) or
            len(np.unique(selected_rows)) != len(selected_rows) or
            np.any(selected_rows < 0) or np.any(selected_rows >= len(sessions))):
        raise ValueError('invalid selected rows')
    mask = np.zeros(len(permutation), dtype=bool)
    mask[selected_rows] = True
    ordered = permutation[mask[permutation]]
    labels = sessions[ordered]
    return [ordered[labels == session] for session in dict.fromkeys(labels)]
