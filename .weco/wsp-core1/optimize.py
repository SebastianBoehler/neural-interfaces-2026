"""Edit the weights of a fitted linear EEG decoder so that it generalises better to unseen people.

The decoder is a 3-class logistic head (motor imagery, mental subtraction, word generation) on 44,239 standardised
features: 51 tangent-space covariance blocks (17 overlapping frequency bands x {full 4 s window, first half, second
half}; each block = 861 entries of a 41-channel log-covariance: 41 channel-power entries and 820 channel-pair entries)
followed by a 328-dimensional slow-waveform block.
"""
import numpy as np


def edit_head(W, inner, stats, meta):
    """Return the edited weight matrix.

    W      (3, D) float32: the fitted head's weights, centred over classes (rows: MI, SUB, WORD).
    inner  (n, 3, D) float32: the same head refitted n times, each with one more training person left out (centred).
    stats  dict of (D,) float32 arrays measured with the inner heads on the person each one never saw:
           'ho'  per-feature contribution to the true-class margin on held-out people,
           'ins' the same contribution in-sample.
    meta   dict of per-feature arrays (length D):
           'block' 0..50 tangent blocks, 51 = slow block; 'band' 0..16 (-1 slow); 'win' 0 full, 1 first half,
           2 second half (-1 slow); 'diag' True for channel-power entries; 'slow' True for the slow block;
           'ch_i', 'ch_j' channel indices of the entry (-1 slow); plus 'bands' (17, 2) band edges in Hz and
           'ch_names' (41,) electrode names.
    Must be deterministic, use only these inputs, and return a finite (3, D) array.
    """
    return W
