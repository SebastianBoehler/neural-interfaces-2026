import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prep import partition, ranks, ordered_streams


def test_person_disjoint_partition_and_order_validation():
    people = np.repeat(np.arange(10), 2)
    tr, val, test = partition(people, held=3, seed=1)
    assert len(set(people[tr])) == 8
    assert len(set(people[val])) == len(set(people[test])) == 1
    assert not set(people[tr]) & set(people[val])
    assert not set(people[tr]) & set(people[test])
    assert not set(people[val]) & set(people[test])
    with pytest.raises(ValueError):
        partition(people, held=99, seed=1)
    with pytest.raises(ValueError):
        ranks(np.array([0, 1, 1]), 3)


def test_ordered_streams_resets_by_stream_and_preserves_permutation():
    stream = np.array(["a", "a", "b", "b", "a", "b"])
    rank = ranks(np.array([4, 3, 0, 5, 2, 1]), 6)
    groups = ordered_streams(np.arange(6), stream, rank)
    assert [x.tolist() for x in groups] == [[4, 0, 1], [3, 5, 2]]
