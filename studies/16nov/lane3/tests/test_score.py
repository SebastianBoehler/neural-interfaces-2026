import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'score.py'
SPEC = importlib.util.spec_from_file_location('lane3_score', PATH)
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def test_validate_fold_guards_person_val_and_probabilities(tmp_path):
    people = np.repeat([f'Dreyer2026Proteus/P{i:02d}' for i in range(1, 11)], 2)
    rows = np.array([0, 1])
    meta = {'arm': 'off', 'seed': 0, 'held': 'P01', 'val_person': 'Dreyer2026Proteus/P02',
            'orders': list(S.ORDERS), 'checkpoint_sha256': S.BLOB, 'trainable': 5022403, 'best_epoch': 0}
    data = {'index': rows, 'metadata': json.dumps(meta), 'val_log': np.ones(25),
            **{f'proba_{o}': np.full((2, 3), 1/3) for o in S.ORDERS}}
    p = tmp_path / 'off_s0_P01.npz'
    np.savez(p, **data)
    assert S.validate_fold(p, 'off', 0, 'P01', people)['within0'].shape == (2, 3)
    data['proba_within0'] = np.ones((2, 3))
    np.savez(p, **data)
    with pytest.raises(ValueError):
        S.validate_fold(p, 'off', 0, 'P01', people)
    data['proba_within0'] = np.full((2, 3), 1/3)
    meta['val_person'] = 'Dreyer2026Proteus/P01'
    data['metadata'] = json.dumps(meta)
    np.savez(p, **data)
    with pytest.raises(ValueError):
        S.validate_fold(p, 'off', 0, 'P01', people)


def test_output_matrix_rejects_missing_and_extra(tmp_path):
    with pytest.raises(ValueError):
        S.matrix_paths(tmp_path)
    for name in S.expected_names():
        (tmp_path / name).touch()
    assert len(S.matrix_paths(tmp_path)) == 60
    (tmp_path / 'extra.npz').touch()
    with pytest.raises(ValueError):
        S.matrix_paths(tmp_path)
