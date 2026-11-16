import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import validate_fold_names  # noqa: E402


def test_collector_refuses_missing_or_extra_person():
    expected = {f'P{i:02d}.npz' for i in range(1, 11)}
    validate_fold_names(expected)
    with pytest.raises(ValueError, match='fold'):
        validate_fold_names(expected - {'P10.npz'})
    with pytest.raises(ValueError, match='fold'):
        validate_fold_names(expected | {'P11.npz'})
