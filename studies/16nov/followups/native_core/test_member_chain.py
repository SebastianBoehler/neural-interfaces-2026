"""Exact zero and unchanged inner-gate guards for native member deltas."""
import numpy as np
import pytest

from member_chain import validate_delta, verify_arm


def records(value):
    return {'b|input': np.ones((2, 3)), 'b|gate': np.ones(2), 'b|V6Fr': np.full((2, 3), value)}


def test_zero_and_legitimate_unchanged_member():
    zero = np.zeros((2, 3))
    validate_delta(zero, 2, strict_zero=True)
    validate_delta(zero, 2, strict_zero=False)
    verify_arm(records(1), records(1), strict_zero=True)
    verify_arm(records(1), records(1), strict_zero=False)
    with pytest.raises(ValueError):
        validate_delta(np.array([[.1, -.1, 0], [0, 0, 0]]), 2, strict_zero=True)
    bad = records(1); bad['b|gate'][0] = 0
    with pytest.raises(ValueError):
        verify_arm(records(1), bad, strict_zero=False)
