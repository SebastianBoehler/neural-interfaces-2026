"""Collect all §300 order shards and run the unchanged saved-stage report."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SYSTEM = HERE.parent / 'system'
sys.path[:0] = [str(HERE), str(SYSTEM)]
import metrics
import shard_guard
from report import report
from rolling_audit import paths_for, record
spec = importlib.util.spec_from_file_location('joint_score', SYSTEM / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, OLD = m.s, m.OLD
NEW = ('E05', 'E05CS', 'E1CS', 'E2CS', 'FBL15', 'EFBL')
OLD_ARMS = ('control', 'E', 'C10S')


def main():
    prereg = ROOT / 'runs/lane1_inputs/attribution-registration.md'
    if not prereg.is_file() or '§300 resource-only sharding amendment' not in prereg.read_text():
        raise ValueError('§300 sharding amendment required')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions)
    m.ARMS = {arm: m.ARMS[arm] for arm in NEW}; new = m.collect_new(len(y), {})
    result = {}
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(ROOT / f'runs/newmember/s281_v66p_zero_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for base in s.BASES: s.verify_zero(old['control', order, base], s.load_output(ROOT, 'zero', order, base, len(y), {}))
        shard = shard_guard.load(ROOT, order, (*OLD_ARMS, *NEW), s.BASES, s, metrics.probability)
        for arm in (*OLD_ARMS, *NEW):
            source_root = OLD if arm in OLD_ARMS else ROOT
            with np.load(source_root / f'runs/chainout/s281_v66p_{arm}_{order}.npz') as z:
                for base in s.BASES:
                    raw = z[f'{base}|V6Fr'].astype(np.float64)
                    if not np.array_equal(raw, shard[f'{arm}|V6Fr|{base}']): raise ValueError('raw shard mismatch')
                    cp = (old if arm in OLD_ARMS else new)[arm, order, base]
                    if not np.array_equal(cp, shard[f'{arm}|CP|{base}']): raise ValueError('CP shard mismatch')
                    for stage in ('V6Fr', 'count', 'CP'): result[arm, stage, order, base] = shard[f'{arm}|{stage}|{base}']
    paths = paths_for(OLD, list(OLD_ARMS)+['zero_eegn'], s.ORDERS, s.BASES)
    paths += paths_for(ROOT, list(NEW)+['zero'], s.ORDERS, s.BASES)
    paths += [p for o in s.ORDERS for p in (ROOT / f'runs/s300_shards/{o}.npz', ROOT / f'runs/s300_shards/{o}.json')]
    paths += [HERE / x for x in ('shard.py','shard_guard.py','collect.py','capture.py','metrics.py','report.py')]
    paths += [SYSTEM / 'score.py', SYSTEM / 'rolling_audit.py', ROOT / 'runs/replay_data/bases.npz']
    audit, sha, previous = record('300', 'saved_stage_attribution', paths, prereg, s.sha256)
    print('§300 exploratory saved-stage attribution; no earlier-stage causal claim.')
    print('all eight shards verified; original CP control input matched in every shard')
    print('audit', audit, 'entry_sha256', sha, 'already_read_blocks', previous, flush=True)
    report(result, y, cells, people, s, metrics, (*OLD_ARMS[1:], *NEW), ('V6Fr', 'count', 'CP'))


if __name__ == '__main__': main()
