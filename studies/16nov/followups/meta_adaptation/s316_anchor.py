"""Validate and load the common §305 anchor for the separate §316 study."""
import numpy as np


def load_common_anchor(shard, meta, cache_hashes, data_sha, perm_sha, source_sha):
    expected = {f'{person}.npz': value for person, value in cache_hashes.items()}
    if (meta['input_sha256'] != expected or meta['data_sha256'] != data_sha or
            meta['permutation_sha256'] != perm_sha or meta['source_sha256'] != source_sha or
            meta['held_accuracy_computed'] is not False):
        raise ValueError('common anchor input provenance mismatch')
    loss = float(meta['native_fit_loss'])
    grad = float(meta['max_gradient'])
    calls = int(meta['objective_evaluations'])
    with np.load(shard, allow_pickle=False) as z:
        coef = np.asarray(z['Nconverged_coef'], dtype=np.float64)
        bias = np.asarray(z['Nconverged_intercept'], dtype=np.float64)
    if (coef.shape != (3, 44239) or bias.shape != (3,) or
            not np.isfinite(coef).all() or not np.isfinite(bias).all() or
            not np.isfinite([loss, grad]).all() or grad > 1e-5 or calls <= 0):
        raise ValueError('invalid common anchor or convergence evidence')
    return np.ascontiguousarray(np.column_stack((coef, bias))), loss, grad, calls
