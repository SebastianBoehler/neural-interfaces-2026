"""Verify the §281 inner gate and zero/nonzero controls before saving."""
import numpy as np


def verify(control, arm, zero):
    assert set(control) == set(arm), "control/arm keys differ"
    bases = {k.rsplit("|", 1)[0] for k in control if k.endswith("|V6Fr")}
    assert bases, "no V6Fr bases"
    assert set(control) == {f"{b}|{k}" for b in bases for k in ("input", "gate", "V6Fr")}, "incomplete gate records"
    for b in sorted(bases):
        assert np.array_equal(control[f"{b}|input"], arm[f"{b}|input"]), f"{b}: gate input changed"
        assert np.array_equal(control[f"{b}|gate"], arm[f"{b}|gate"]), f"{b}: gate decisions changed"
        same = np.array_equal(control[f"{b}|V6Fr"], arm[f"{b}|V6Fr"])
        assert same if zero else not same, f"{b}: zero arm changed" if zero else f"{b}: nonzero arm did not change"


BASES = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")


def verify_archive(outputs, archive):
    """Compare the corrected control against the archived §257 chain."""
    for base in BASES:
        for arm in ("P0r", "V6Fr"):
            key = f"{base}|{arm}"
            assert key in outputs, f"control missing {key}"
            assert key in archive, f"archive missing {key}"
            x, y = outputs[key], archive[key]
            assert x.shape == y.shape and x.dtype == y.dtype, f"{key}: shape or dtype differs"
            assert np.isfinite(x).all() and np.isfinite(y).all(), f"{key}: non-finite output"
            if x.tobytes() != y.tobytes():
                diff = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                labels = int((x.argmax(axis=-1) != y.argmax(axis=-1)).sum())
                raise AssertionError(f"{key}: archive mismatch, max abs {diff:.3g}, labels changed {labels}")
