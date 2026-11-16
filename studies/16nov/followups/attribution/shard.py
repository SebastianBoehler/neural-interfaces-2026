"""Generate one §300 order's original count-filter outputs; no metrics."""
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
from rolling_audit import paths_for
spec = importlib.util.spec_from_file_location('joint_score', SYSTEM / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, OLD = m.s, m.OLD
NEW = ('E05', 'E05CS', 'E1CS', 'E2CS', 'FBL15', 'EFBL')
OLD_ARMS = ('control', 'E', 'C10S')


def main(order, smoke=False):
    if order not in s.ORDERS: raise ValueError('unknown pinned order')
    prereg = ROOT / 'runs/lane1_inputs/attribution-registration.md'
    if not prereg.is_file() or '### 300.' not in prereg.read_text(): raise ValueError('§300 registration required')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        n = len(z['y|all_loso']); sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, n, sessions)
    m.ARMS = {arm: m.ARMS[arm] for arm in NEW}; new = m.collect_new(n, {})
    for o in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{o}_gate.npz') as c, \
             np.load(ROOT / f'runs/newmember/s281_v66p_zero_{o}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for b in s.BASES: s.verify_zero(old['control', o, b], s.load_output(ROOT, 'zero', o, b, n, {}))
    original = capture.original_filtered(OLD, 'within0', 'B2_pool')
    import package as K
    names = ('package', 'jointcount', 'common', 'detect')
    actual = [Path(x.__file__).resolve(strict=True) for x in (K, K.J, K.C, K.D)]
    nominal = [ROOT / f'studies/14nov/rundetect/{name}.py' for name in names]
    for a, b in zip(actual, nominal):
        if s.sha256(a) != s.sha256(b): raise ValueError('actual filter source mismatch')
    # Directly validate the captured original control construction on every shard.
    with np.load(OLD / 'runs/chainout/s281_v66p_control_within0.npz') as z: raw0 = z['B2_pool|V6Fr'].astype(np.float64)
    hc0 = K.J.hyp_cache('within0', 'wprior', K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    filt0 = K.gate(raw0, K.J.filt_joint(raw0, hc0, 'urn'), 'within0', K.latch('within0'))
    if not np.array_equal(filt0, original): raise ValueError('original CP input mismatch')
    lat = K.latch(order); hc = K.J.hyp_cache(order, 'wprior', K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    out = {}
    for arm in (('control', 'E05CS') if smoke else (*OLD_ARMS, *NEW)):
        root = OLD if arm in OLD_ARMS else ROOT
        with np.load(root / f'runs/chainout/s281_v66p_{arm}_{order}.npz') as chain, \
             np.load(root / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz') as gate:
            for base in (('B2_pool',) if smoke else s.BASES):
                raw = chain[f'{base}|V6Fr'].astype(np.float64)
                if not np.array_equal(raw, gate[f'{base}|V6Fr']): raise ValueError('gate/chain mismatch')
                metrics.probability(raw)
                filtered = K.gate(raw, K.J.filt_joint(raw, hc, 'urn'), order, lat)
                metrics.probability(filtered)
                for stage, values in (('V6Fr', raw), ('count', filtered), ('CP', (old if arm in OLD_ARMS else new)[arm, order, base])):
                    out[f'{arm}|{stage}|{base}'] = values
    paths = paths_for(OLD, list(OLD_ARMS) + ['zero_eegn'], (order,), s.BASES)
    paths += paths_for(ROOT, list(NEW) + ['zero'], (order,), s.BASES)
    paths += actual + nominal + [HERE / x for x in ('shard.py', 'capture.py', 'metrics.py', 'report.py')]
    paths += [SYSTEM / 'score.py', SYSTEM / 'rolling_audit.py', OLD / 'studies/14nov/newmember/s281_cp.py',
              ROOT / 'studies/14nov/newmember/score_s_v66p.py', ROOT / 'runs/replay_data/bases.npz',
              ROOT / 'runs/shuffle_perms.npz', ROOT / 'runs/sealed/final_zips/v66_cp1_cap250.zip', ROOT / 'data/proteus_windows.npz', prereg]
    folder = 's300_shards_smoke' if smoke else 's300_shards'
    dst = ROOT / f'runs/{folder}/{order}.npz'; side = dst.with_suffix('.json')
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or side.exists(): raise ValueError('refusing existing shard output')
    with dst.open('xb') as f: np.savez_compressed(f, **out)
    manifest = {'order': order, 'smoke': smoke, 'sha256': s.sha256(dst), 'keys': sorted(out),
                'input_sha256': {str(Path(p).resolve(strict=True)): s.sha256(p) for p in sorted(set(paths))}}
    with side.open('x') as f: json.dump(manifest, f, sort_keys=True)
    print(f'OK {order}: {len(out)} checked stage arrays, no metrics', flush=True)


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != '--smoke'): raise ValueError('invalid arguments')
    main(sys.argv[1], smoke=len(sys.argv) == 3)
