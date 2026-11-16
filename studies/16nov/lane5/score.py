"""Complete-matrix native Lane 5 scoring using the existing official helpers."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('NI26_WS', Path(__file__).resolve().parents[3])).resolve()
sys.path.insert(0, str(ROOT / 'studies/16nov/lane5'))
from run import ARMS, ORDERS, sha, validate_fit_status  # noqa: E402
from bridge import Native  # noqa: E402

PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))
MODES = ('static', 'adaptive')


def validate_status(entry, n):
    a, b = entry['static'], entry['adaptive']
    expected = len(range(40, n, 20))
    if (a['windows'] != n or a['updates'] != 0 or b['windows'] != n or b['updates'] != expected or
            not np.isfinite([b['max_grad'], b['delta']]).all() or b['delta'] > .100001 or
            (expected and (b['max_grad'] <= 0 or b['delta'] <= 0))):
        raise ValueError(f'invalid adaptation status: {entry}')
    if expected and b['prefix_count'] != min(64, 40 + 20 * (expected - 1)):
        raise ValueError('wrong adaptation prefix count')


def check_manifest(root, path, lib):
    records = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name in records or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'invalid or duplicate manifest entry {name}')
        records[name] = digest
    required = {'data/proteus_windows.npz', 'runs/shuffle_perms.npz', 'experiments/protocol.py',
                'studies/16nov/lane5/bridge.py', 'studies/16nov/lane5/run.py',
                'studies/16nov/lane5/score.py', 'studies/16nov/lane2/score.py',
                'studies/14nov/newmember/score_s_v66p.py', 'studies/14nov/newmember/s281_provenance.py',
                'runs/lane4_reference/reference_sha256.json'}
    required.add(str(lib.relative_to(root)))
    required.update(f'studies/16nov/lane5/native/{x}' for x in
                    ('api.h', 'internal.h', 'model.cpp', 'train.cpp', 'session.cpp', 'api.cpp', 'build.sh'))
    if not required <= set(records):
        raise ValueError(f'manifest missing required inputs: {sorted(required - set(records))}')
    for name, digest in records.items():
        if sha(root / name) != digest:
            raise ValueError(f'manifest mismatch: {name}')
    return records


def helpers():
    spec = importlib.util.spec_from_file_location('lane2_score', ROOT / 'studies/16nov/lane2/score.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ORDERS, module.REF = ORDERS, ROOT / 'runs/lane4_reference'
    return module


def without_labels(root):
    with np.load(root / 'data/proteus_windows.npz', allow_pickle=False) as z:
        d = {key: z[key].copy() for key in ('subject', 'session', 'interface', 'split')}
    d['stream'] = np.char.add(np.char.add(d['subject'], '|'), d['session'])
    return d


def load_outputs(d, manifest, library):
    n = len(d['subject']); path = ROOT / 'runs/lane5'
    expected = {f'{arm}_s{seed}_{held}.npz' for arm, _, _ in ARMS for seed in range(2) for held in PEOPLE}
    found = {p.name for p in path.glob('*.npz')}
    if found != expected:
        raise ValueError(f'wrong output matrix: missing={sorted(expected-found)} extra={sorted(found-expected)}')
    checkpoints = ROOT / 'runs/lane5_checkpoints'
    expected_cp = {name[:-4] + '.pt' for name in expected}
    found_cp = {p.name for p in checkpoints.glob('*.pt')}
    if found_cp != expected_cp:
        raise ValueError(f'wrong checkpoint matrix: missing={sorted(expected_cp-found_cp)} extra={sorted(found_cp-expected_cp)}')
    names = {a: (arch, consistency) for a, arch, consistency in ARMS}
    hashes, fit_hash, result = {}, {}, {}
    native = Native(library)
    folds = {p: (tr, te) for p, tr, te in helpers().protocol.folds(d, 'all_loso')}
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        permutations = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    for o, p in permutations.items():
        if not np.array_equal(np.sort(p), np.arange(n)):
            raise ValueError(f'{o}: invalid global permutation')
    mh, lh = sha(manifest), sha(library)
    dh, ph = sha(ROOT / 'data/proteus_windows.npz'), sha(ROOT / 'runs/shuffle_perms.npz')
    for arm, (arch, consistency) in names.items():
        for seed in range(2):
            for held in PEOPLE:
                tr, te = folds[held]; file = path / f'{arm}_s{seed}_{held}.npz'; before = sha(file)
                with np.load(file, allow_pickle=False) as z:
                    required = {'metadata'} | {f'{o}|{k}' for o in ORDERS for k in ('index', *MODES)}
                    if set(z.files) != required:
                        raise ValueError(f'{file}: wrong output fields')
                    m = json.loads(str(z['metadata'].item()))
                    identity = (m.get('arm'), m.get('architecture'), m.get('consistency'), m.get('seed'), m.get('held'))
                    if identity != (arm, arch, consistency, seed, held) or m['epochs'] != 60:
                        raise ValueError(f'{file}: wrong model or fold')
                    if set(m['fit_people']) != set(d['subject'][tr]) or set(m['fit_people']) & set(d['subject'][te]):
                        raise ValueError(f'{file}: training-person mismatch')
                    if (m['library_sha256'] != lh or m['manifest_sha256'] != mh or
                            m['data_sha256'] != dh or m['permutations_sha256'] != ph):
                        raise ValueError(f'{file}: provenance mismatch')
                    cp = checkpoints / f'{arm}_s{seed}_{held}.pt'
                    if m['checkpoint_path'] != str(cp.relative_to(ROOT)) or sha(cp) != m['checkpoint_sha256']:
                        raise ValueError(f'{file}: checkpoint path or checksum mismatch')
                    model = native.create(arch, 0, seed)
                    model.load(cp)
                    if hashlib.sha256(model.params().tobytes()).hexdigest() != m['parameter_sha256']:
                        raise ValueError(f'{file}: checkpoint parameter hash mismatch')
                    model.close()
                    hashes[cp.name] = m['checkpoint_sha256']
                    f = m['fit_status']
                    validate_fit_status(f, arch, consistency)
                    fit_hash[arm, seed, held] = f['shuffle_hash']
                    for o, perm in permutations.items():
                        rows = perm[te[perm]]
                        if m['order_sha'][o] != hashlib.sha256(perm.tobytes()).hexdigest() or not np.array_equal(z[f'{o}|index'], rows):
                            raise ValueError(f'{file}: order mismatch {o}')
                        arrays = {mode: z[f'{o}|{mode}'] for mode in MODES}
                        for mode, a in arrays.items():
                            if a.shape != (len(rows), 3) or not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(1), 1, atol=1e-6):
                                raise ValueError(f'{file}: invalid {o}/{mode}')
                            result.setdefault((arm, seed, o, mode), np.full((n, 3), np.nan))[rows] = a
                        entries = m['status'][o]
                        sessions = set(d['stream'][rows])
                        if len(entries) != len(sessions) or {x['session'] for x in entries} != sessions:
                            raise ValueError(f'{file}: invalid session coverage')
                        for entry in entries:
                            loc = np.flatnonzero(d['stream'][rows] == entry['session'])
                            validate_status(entry, len(loc))
                            if not np.allclose(arrays['static'][loc[:40]], arrays['adaptive'][loc[:40]], atol=1e-6, rtol=1e-6):
                                raise ValueError(f'{file}: adaptation changed a pre-update window')
                if sha(file) != before:
                    raise ValueError(f'{file}: changed while reading')
                hashes[file.name] = before
    for seed in range(2):
        for held in PEOPLE:
            if fit_hash['hybrid_ce', seed, held] != fit_hash['hybrid_consistency', seed, held]:
                raise ValueError(f'hybrid seed/order mismatch: {held} s{seed}')
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('unfilled prediction rows')
    return result, hashes


def main():
    manifest = Path(os.environ['LANE5_MANIFEST']).resolve(strict=True)
    library = Path(os.environ['LANE5_LIB']).resolve(strict=True)
    if sha(manifest) != os.environ['LANE5_MANIFEST_SHA']:
        raise ValueError('manifest lock mismatch')
    check_manifest(ROOT, manifest, library)
    d = without_labels(ROOT)
    result, hashes = load_outputs(d, manifest, library)
    s = helpers(); n = len(d['subject']); reference = s.check_reference(n)
    # Only after all predictions, provenance, controls and reference pass do we load held labels.
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        y = z['y'].copy()
    person = np.array([x.split('/')[-1] for x in d['subject']])
    cell = np.char.add(np.char.add(d['stream'], '|'), d['interface'])
    if any(set(y[cell == c]) != {0, 1, 2} for c in np.unique(cell)):
        raise ValueError('cell lacks a class required for official/helper metric parity')
    pp, official = {}, {}
    models = [(a, mode) for a, _, _ in ARMS for mode in MODES] + [('reference', 'static')]
    for arm, mode in models:
        for o in ORDERS:
            preds = ([reference[o, base] for base in s.BASES] if arm == 'reference' else
                     [result[arm, seed, o, mode] for seed in range(2)])
            pp[arm, mode, o] = np.mean([s.person_accuracy(x, y, cell, person, PEOPLE) for x in preds], axis=0)
            official[arm, mode, o] = float(np.mean([100 * s.protocol.score(y, x.argmax(1), d, np.ones(n, bool))['balanced_accuracy_cell_mean'] for x in preds]))
    def summary(a, ma, b, mb):
        return s.summarize(np.stack([pp[a, ma, o] - pp[b, mb, o] for o in ORDERS]), ORDERS, PEOPLE)
    comparisons = {(f'{a}_adaptive-minus-static'): summary(a, 'adaptive', a, 'static') for a, _, _ in ARMS}
    for mode in MODES:
        comparisons[f'hybrid_consistency-minus-ce_{mode}'] = summary('hybrid_consistency', mode, 'hybrid_ce', mode)
        comparisons[f'hybrid-minus-connectivity_{mode}'] = summary('hybrid_ce', mode, 'connectivity_ce', mode)
        for arm, _, _ in ARMS:
            comparisons[f'{arm}_{mode}-minus-reference'] = summary(arm, mode, 'reference', 'static')
    report = {'official_equal_cell_by_order': {f'{a}_{m}': {o: official[a, m, o] for o in ORDERS} for a, m in models},
              'official_equal_cell_scheme_equal': {f'{a}_{m}': float(np.mean([
                  np.mean([official[a, m, o] for o in ORDERS if o.startswith(scheme)])
                  for scheme in ('within', 'full', 'runs')])) for a, m in models},
              'absolute_people_order': {f'{a}_{m}': s.summarize(
                  np.stack([pp[a, m, o] for o in ORDERS]), ORDERS, PEOPLE) for a, m in models},
              'people_order_intervals': comparisons, 'output_sha256': hashes,
              'manifest_sha256': sha(manifest), 'library_sha256': sha(library),
              'seed_aggregation': 'argmax and score each seed separately, then average metrics',
              'reference': 'validated new-person v6.6 route emulation; not returning-person validation',
              'p04_p10': 'held-out new-person folds, not returning sessions'}
    target = ROOT / 'evidence/next_wave_16nov/lane5_results.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True); f.write('\n')
    print(f'validated full matrix and saved {target}')


if __name__ == '__main__':
    main()
