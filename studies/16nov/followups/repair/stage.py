"""Apply the existing §300 stage report to complete §302 outputs."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ATTR = HERE.parent / 'attribution'
sys.path.insert(0, str(ATTR))
import capture
import metrics


def report_stages(root, oldroot, old, new, y, cells, people, s, prereg):
    original = capture.original_filtered(oldroot, 'within0', 'B2_pool')
    import package as K
    names = ('package', 'jointcount', 'common', 'detect')
    actual = [Path(module.__file__).resolve(strict=True) for module in (K, K.J, K.C, K.D)]
    for module, name in zip((K, K.J, K.C, K.D), names):
        nominal = root / f'studies/14nov/rundetect/{name}.py'
        if s.sha256(Path(module.__file__).resolve()) != s.sha256(nominal):
            raise ValueError('actual count-filter source differs from pinned copy')
    scores = {}
    arms = ('control', 'E05', 'ET05', 'ET2', 'EPR25', 'EPR50')
    for order in s.ORDERS:
        lat = K.latch(order)
        hc = K.J.hyp_cache(order, 'wprior', K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
        for arm in arms:
            source_root = oldroot if arm == 'control' else root
            with np.load(source_root / f'runs/chainout/s281_v66p_{arm}_{order}.npz') as chain, \
                 np.load(source_root / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz') as gate:
                for base in s.BASES:
                    raw = chain[f'{base}|V6Fr'].astype(np.float64)
                    if not np.array_equal(raw, gate[f'{base}|V6Fr']): raise ValueError('gate/chain differs')
                    metrics.probability(raw)
                    filtered = K.gate(raw, K.J.filt_joint(raw, hc, 'urn'), order, lat)
                    metrics.probability(filtered)
                    if (arm, order, base) == ('control', 'within0', 'B2_pool') and not np.array_equal(filtered, original):
                        raise ValueError('original CP control input differs')
                    scores[arm, 'V6Fr', order, base] = raw
                    scores[arm, 'count', order, base] = filtered
                    scores[arm, 'CP', order, base] = (old if arm == 'control' else new)[arm, order, base]
    # The report algorithm is unchanged; only its two named stage-contrast arms change to the registered repairs.
    path = ATTR / 'report.py'; source = path.read_text()
    old_arm_loop = "    for arm in ('E05', 'E1CS'):\n"
    new_arm_loop = "    for arm in ('ET05', 'ET2', 'EPR25', 'EPR50'):\n"
    if source.count(old_arm_loop) != 1: raise ValueError('unexpected §300 report source')
    namespace = {'__name__': 'repair_stage_report', '__file__': str(path)}
    exec(compile(source.replace(old_arm_loop, new_arm_loop, 1), str(path), 'exec'), namespace)
    from rolling_audit import paths_for, record
    paths = paths_for(root, list(arms[1:]) + ['ZR'], s.ORDERS, s.BASES)
    paths += paths_for(oldroot, ['control', 'zero_eegn'], s.ORDERS, s.BASES)
    paths += actual + [root / f'studies/14nov/rundetect/{name}.py' for name in names]
    paths += [HERE / 'stage.py', ATTR / 'capture.py', ATTR / 'metrics.py', ATTR / 'report.py']
    audit, sha, previous = record('302', 'saved_stage_diagnostics', paths, prereg, s.sha256)
    print('§302 saved stages versus control; original CP input matched bitwise; no earlier-stage causal claim')
    print('audit', audit, 'entry_sha256', sha, 'already_read_blocks', previous, flush=True)
    namespace['report'](scores, y, cells, people, s, metrics, arms[1:], ('V6Fr', 'count', 'CP'))
    for stage in ('V6Fr', 'count', 'CP'):
        for order in s.ORDERS:
            for base in s.BASES:
                scores['control', stage, order, base] = scores['E05', stage, order, base]
    print('§302 saved stages versus E05; no earlier-stage causal claim', flush=True)
    namespace['report'](scores, y, cells, people, s, metrics, arms[2:], ('V6Fr', 'count', 'CP'))
