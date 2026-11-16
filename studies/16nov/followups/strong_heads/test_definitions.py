import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from definitions import ARMS, ORDERS, SEED_ARMS


def test_complete_registered_seed_matrix():
    assert len(ARMS) == 5 and len(ORDERS) == 8
    assert len(SEED_ARMS) == 11 and SEED_ARMS[0] == 'Rzero'
    assert set(SEED_ARMS[1:]) == {f'{arm}{seed}' for arm in ARMS for seed in (0, 1)}
