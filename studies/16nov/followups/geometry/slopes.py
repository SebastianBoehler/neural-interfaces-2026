"""Exact effects of scalar SPD congruence on the frozen LR and PCA views."""
import numpy as np


def coefficients(head_weights, head_scale, pca_components, pca_scale, channels=41, bands=17, blocks=3):
    head_weights, head_scale, pca_components, pca_scale = map(
        lambda x: np.asarray(x, dtype=np.float64), (head_weights, head_scale, pca_components, pca_scale))
    if (head_weights.ndim != 2 or head_weights.shape[0] != 3 or pca_components.ndim != 2 or
            head_weights.shape[1] != pca_components.shape[1] or
            head_scale.shape != (head_weights.shape[1],) or pca_scale.shape != head_scale.shape or
            (head_scale <= 0).any() or (pca_scale <= 0).any() or
            any(not np.isfinite(x).all() for x in (head_weights, head_scale, pca_components, pca_scale))):
        raise ValueError('invalid head/PCA weights or scaler')
    width = channels * (channels + 1) // 2
    if len(head_scale) < bands * blocks * width:
        raise ValueError('feature layout is smaller than the tangent blocks')
    i, j = np.triu_indices(channels)
    diagonal = np.flatnonzero(i == j)
    core, view = np.empty((3, bands)), np.empty((len(pca_components), bands))
    for b in range(bands):
        indices = np.concatenate([(b * blocks + k) * width + diagonal for k in range(blocks)])
        core[:, b] = (head_weights[:, indices] / head_scale[indices]).sum(axis=1)
        view[:, b] = (pca_components[:, indices] / pca_scale[indices]).sum(axis=1)
    return core, view
