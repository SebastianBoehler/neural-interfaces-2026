"""Change label composition of disjoint prefixes while holding the source row pool fixed."""
import numpy as np


def episodes(labels, streams, seed, steps):
    rng = np.random.default_rng(seed)
    sessions = [[np.flatnonzero((streams == s) & (labels == c)) for c in range(3)]
                for s in np.unique(streams)]
    for _ in range(steps):
        groups = sessions[rng.integers(len(sessions))]
        a, b, c = [rng.choice(groups[k], 24, replace=False) for k in rng.permutation(3)]
        skew = np.stack([rng.permutation(np.r_[a[:16], c[:8]]),
                         rng.permutation(np.r_[b[:16], c[8:16]])])
        random = rng.permutation(skew.ravel()).reshape(2, 24)
        query = rng.permutation(np.r_[a[16:], b[16:], c[16:]])
        assert len(np.unique(np.r_[skew.ravel(), query])) == 72
        assert np.array_equal(np.sort(skew.ravel()), np.sort(random.ravel()))
        assert np.array_equal(np.bincount(labels[query]), [8, 8, 8])
        yield skew, random, query
