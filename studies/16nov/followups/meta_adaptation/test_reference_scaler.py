"""The target anchor must use the verified §305 reference scaler."""
import hashlib
import json

import numpy as np
import pytest

from reference_scaler import load_scaler


def test_verified_scaler_and_training_rows(tmp_path):
    path = tmp_path / 'P01.npz'
    rows = np.array([1, 3], dtype=np.int64)
    np.savez(path, fit_rows=rows, mean=np.array([1., 2.]), scale=np.array([3., 4.]))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = {'files': {'P01.npz': sha}, 'held_accuracy_computed': False}
    (tmp_path / 'manifest.json').write_text(json.dumps(lock))
    mean, scale = load_scaler(tmp_path, 'P01', rows, sha, 2)
    np.testing.assert_array_equal(mean, [1., 2.])
    np.testing.assert_array_equal(scale, [3., 4.])
    with pytest.raises(ValueError):
        load_scaler(tmp_path, 'P01', rows[::-1], sha, 2)
    with pytest.raises(ValueError):
        load_scaler(tmp_path, 'P01', rows, '0' * 64, 2)
