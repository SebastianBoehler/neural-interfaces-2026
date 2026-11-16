"""Ensure the native scorer retains the existing guarded collector."""
from pathlib import Path

import pytest

from score import patch_score


def test_native_score_reuses_guarded_collector():
    root = Path(__file__).resolve().parents[4]
    source = (root / 'studies/16nov/followups/geometry/score.py').read_text()
    result = patch_score(source)
    assert 'def scorer():' in result and 'def verify_geometry(' in result
    assert "'Rzero'" in result and "('Ntrain', 'Nfixed')" in result
    compile(result, '<native-score>', 'exec')
    with pytest.raises(ValueError):
        patch_score(result)
