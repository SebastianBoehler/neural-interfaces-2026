"""§300: locate changes at saved chain, count-filter and CP outputs."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SYSTEM = HERE.parent / 'system'
sys.path[:0] = [str(HERE), str(SYSTEM)]
import capture
import metrics
from rolling_audit import paths_for, record
spec = importlib.util.spec_from_file_location('joint_score', SYSTEM / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, OLD = m.s, m.OLD
NEW = ('E05', 'E05CS', 'E1CS', 'E2CS', 'FBL15', 'EFBL')
OLD_ARMS = ('control', 'E', 'C10S')
STAGES = ('V6Fr', 'count', 'CP')


def main():
    prereg = ROOT / 'runs/lane1_inputs/attribution-registration.md'
    if not prereg.is_file() or '### 300.' not in prereg.read_text(): raise ValueError('§300 registration required')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions)
    m.ARMS = {arm: m.ARMS[arm] for arm in NEW}
    new = m.collect_new(len(y), {})
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(ROOT / f'runs/newmember/s281_v66p_zero_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for base in s.BASES:
            s.verify_zero(old['control', order, base], s.load_output(ROOT, 'zero', order, base, len(y), {}))
    # Execute the pinned original CP code with a read-only capture immediately before personalisation.
    original = capture.original_filtered(OLD, 'within0', 'B2_pool')
    import package as K
    modules = (K, K.J, K.C, K.D)
    actual = [Path(module.__file__).resolve(strict=True) for module in modules]
    nominal = [ROOT / f'studies/14nov/rundetect/{name}.py' for name in ('package', 'jointcount', 'common', 'detect')]
    for observed, expected in zip(actual, nominal):
        if s.sha256(observed) != s.sha256(expected):
            raise ValueError(f'actual count-filter source differs: {observed} vs {expected}')
    scores = {}
    for order in s.ORDERS:
        lat = K.latch(order)
        hc = K.J.hyp_cache(order, 'wprior', K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
        for arm in (*OLD_ARMS, *NEW):
            source_root = OLD if arm in OLD_ARMS else ROOT
            path = source_root / f'runs/chainout/s281_v66p_{arm}_{order}.npz'
            audit = source_root / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz'
            with np.load(path) as chain, np.load(audit) as gate:
                for base in s.BASES:
                    raw = chain[f'{base}|V6Fr'].astype(np.float64)
                    if not np.array_equal(raw, gate[f'{base}|V6Fr']): raise ValueError('gate/chain output differs')
                    metrics.probability(raw)
                    filtered = K.gate(raw, K.J.filt_joint(raw, hc, 'urn'), order, lat)
                    metrics.probability(filtered)
                    if arm == 'control' and order == 'within0' and base == 'B2_pool':
                        if not np.array_equal(filtered, original): raise ValueError('original CP filtered input differs')
                    scores[arm, 'V6Fr', order, base] = raw
                    scores[arm, 'count', order, base] = filtered
                    scores[arm, 'CP', order, base] = (old if arm in OLD_ARMS else new)[arm, order, base]
    paths = paths_for(OLD, list(OLD_ARMS) + ['zero_eegn'], s.ORDERS, s.BASES)
    paths += paths_for(ROOT, list(NEW) + ['zero'], s.ORDERS, s.BASES)
    paths += actual + nominal
    paths += [HERE / name for name in ('run_v2.py', 'capture.py', 'metrics.py', 'report.py')]
    paths += [SYSTEM / name for name in ('score.py', 'rolling_audit.py')]
    paths += [ROOT / 'studies/14nov/newmember/score_s_v66p.py']
    paths += [OLD / 'studies/14nov/newmember/s281_cp.py', ROOT / 'runs/replay_data/bases.npz',
              ROOT / 'runs/shuffle_perms.npz', ROOT / 'runs/sealed/final_zips/v66_cp1_cap250.zip', ROOT / 'data/proteus_windows.npz']
    audit, sha, previous = record('300', 'saved_stage_attribution', paths, prereg, s.sha256)
    print('§300 exploratory saved-stage attribution; no earlier-stage causal claim.')
    print('original CP count-filter input matched bitwise for control/within0/B2_pool')
    print('audit', audit, 'entry_sha256', sha, 'already_read_blocks', previous)
    from report import report
    report(scores, y, cells, people, s, metrics, (*OLD_ARMS[1:], *NEW), STAGES)


if __name__ == '__main__': main()
