"""Validate the complete §303 matrix, then score with the existing §293 metric."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path.insert(0, str(ROOT / 'studies/16nov/lane5'))
from run import ARMS, ORDERS, sha  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


old = module('old_score', ROOT / 'studies/16nov/lane5/score.py')
new = module('repair_run', ROOT / 'studies/16nov/lane5_repair/run.py')
MODES = ('static', 'adaptive', 'repair', 'reverse')
PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))


def load_complete(d, manifest, library, extension, original):
    path = ROOT / 'runs/repair'; n = len(d['subject'])
    expected = {f'{arm}_s{seed}_{held}.npz' for arm, _, _ in ARMS for seed in range(2) for held in PEOPLE}
    found = {p.name for p in path.glob('*.npz')}
    if found != expected:
        raise ValueError(f'incomplete matrix: missing={sorted(expected-found)} extra={sorted(found-expected)}')
    folds = {p: (tr, te) for p, tr, te in old.helpers().protocol.folds(d, 'all_loso')}
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    if any(not np.array_equal(np.sort(p), np.arange(n)) for p in perms.values()):
        raise ValueError('invalid order permutation')
    records = json.loads((ROOT / 'frozen/records.json').read_text())
    sums = {'manifest_sha256': sha(manifest), 'library_sha256': sha(library),
            'extension_sha256': sha(extension), 'data_sha256': sha(ROOT / 'data/proteus_windows.npz'),
            'permutations_sha256': sha(ROOT / 'runs/shuffle_perms.npz')}
    result, hashes, tie_counts, diagnostics = {}, {}, {}, []
    for arm, architecture, _ in ARMS:
        for seed in range(2):
            for held in PEOPLE:
                file = path / f'{arm}_s{seed}_{held}.npz'; digest = sha(file); tr, te = folds[held]
                checkpoint = ROOT / f'runs/lane5_checkpoints/{arm}_s{seed}_{held}.pt'
                reference = ROOT / f'runs/lane5/{arm}_s{seed}_{held}.npz'
                with np.load(file, allow_pickle=False) as z:
                    required = {'metadata'} | {f'{o}|{k}' for o in ORDERS for k in ('index', *MODES)}
                    if set(z.files) != required:
                        raise ValueError(f'{file}: wrong fields')
                    meta = json.loads(str(z['metadata'].item()))
                    if (meta['arm'], meta['architecture'], meta['seed'], meta['held']) != (arm, architecture, seed, held):
                        raise ValueError(f'{file}: identity mismatch')
                    if set(meta['fit_people']) != set(d['subject'][tr]) or set(meta['fit_people']) & set(d['subject'][te]):
                        raise ValueError(f'{file}: fit-person mismatch')
                    if any(meta.get(k) != v for k, v in sums.items()):
                        raise ValueError(f'{file}: input provenance mismatch')
                    if (meta['checkpoint_sha256'] != records[checkpoint.name] or sha(checkpoint) != records[checkpoint.name] or
                            meta['reference_sha256'] != records[reference.name] or sha(reference) != records[reference.name]):
                        raise ValueError(f'{file}: checkpoint/reference mismatch')
                    with np.load(reference, allow_pickle=False) as prior:
                        previous = json.loads(str(prior['metadata'].item()))
                    gpu = previous['gpu_name']
                    if 'RTX 2080 Ti' in gpu:
                        policy, cudnn_tf32 = 'original_2080_a100_cudnn_fp32', False
                    elif 'A100' in gpu:
                        policy, cudnn_tf32 = 'original_a100_default', True
                    else:
                        raise ValueError(f'{file}: unknown original GPU')
                    flags = meta['flags_after']
                    if (meta['original_gpu'] != gpu or 'A100' not in meta['gpu_name'] or
                            meta['precision_policy'] != policy or flags['cudnn_tf32'] is not cudnn_tf32 or
                            flags['matmul_tf32'] is not False or meta['torch_version'] != previous['torch_version'] or
                            meta['torch_cuda'] != previous['torch_cuda']):
                        raise ValueError(f'{file}: precision or runtime provenance mismatch')
                    if set(meta['orders']) != set(ORDERS) or set(meta['status']) != set(ORDERS):
                        raise ValueError(f'{file}: incomplete orders')
                    for order, perm in perms.items():
                        rows = perm[te[perm]]
                        if (meta['order_sha'][order] != hashlib.sha256(perm.tobytes()).hexdigest() or
                                not np.array_equal(z[f'{order}|index'], rows)):
                            raise ValueError(f'{file}: order mismatch {order}')
                        arrays = {k: z[f'{order}|{k}'] for k in MODES}
                        for k, a in arrays.items():
                            if a.shape != (len(rows), 3) or not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(1), 1, atol=1e-6):
                                raise ValueError(f'{file}: invalid {order}/{k}')
                            result.setdefault((arm, seed, order, k), np.full((n, 3), np.nan))[rows] = a
                        for k in ('static', 'adaptive'):
                            check = new.verify_legacy(arrays[k], original[arm, seed, order, k][rows], f'{file}/{order}/{k}')
                            recorded = meta['legacy_checks'][order][k]
                            if check != recorded:
                                raise ValueError(f'{file}: legacy check metadata differs')
                            tie_counts[k] = tie_counts.get(k, 0) + check['near_ties']
                        entries = meta['status'][order]; sessions = set(d['stream'][rows])
                        if len(entries) != len(sessions) or {x['session'] for x in entries} != sessions:
                            raise ValueError(f'{file}: session coverage differs')
                        for entry in entries:
                            loc = np.flatnonzero(d['stream'][rows] == entry['session'])
                            for k in ('repair', 'reverse'):
                                new.check_status(entry[k], len(loc))
                                if not np.allclose(arrays['static'][loc[:40]], arrays[k][loc[:40]], atol=1e-7, rtol=0):
                                    raise ValueError(f'{file}: {k} changed before first update')
                                diagnostics.append((arm, k, entry[k]['attempts'], entry[k]['updates'], entry[k]['skips'], entry[k]['selected_total']))
                if sha(file) != digest:
                    raise ValueError(f'{file}: changed while reading')
                hashes[file.name] = digest
    if any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError('incomplete prediction rows')
    return result, hashes, tie_counts, diagnostics


def main():
    manifest = Path(os.environ['REPAIR_MANIFEST']).resolve(strict=True)
    library = Path(os.environ['LANE5_LIB']).resolve(strict=True)
    extension = Path(os.environ['LANE5_REPAIR_LIB']).resolve(strict=True)
    if sha(manifest) != os.environ['REPAIR_MANIFEST_SHA']:
        raise ValueError('repair manifest mismatch')
    old.check_manifest(ROOT, ROOT / 'source.sha256', library)
    d = old.without_labels(ROOT)
    original, original_hashes = old.load_outputs(d, ROOT / 'source.sha256', library)
    result, hashes, ties, diagnostics = load_complete(d, manifest, library, extension, original)
    # No held labels are loaded before both complete matrices and all guards pass.
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        y = z['y'].copy()
    person = np.array([x.split('/')[-1] for x in d['subject']])
    cell = np.char.add(np.char.add(d['stream'], '|'), d['interface'])
    if any(set(y[cell == c]) != {0, 1, 2} for c in np.unique(cell)):
        raise ValueError('cell lacks a class')
    s = old.helpers(); pp, official = {}, {}
    for arm, _, _ in ARMS:
        for mode in MODES:
            for order in ORDERS:
                preds = [result[arm, seed, order, mode] for seed in range(2)]
                pp[arm, mode, order] = np.mean([s.person_accuracy(x, y, cell, person, PEOPLE) for x in preds], axis=0)
                official[arm, mode, order] = float(np.mean([100 * s.protocol.score(y, x.argmax(1), d, np.ones(len(y), bool))['balanced_accuracy_cell_mean'] for x in preds]))
    def summary(arm, left, right):
        return s.summarize(np.stack([pp[arm, left, o] - pp[arm, right, o] for o in ORDERS]), ORDERS, PEOPLE)
    contrasts = {f'{arm}_{left}-minus-{right}': summary(arm, left, right)
                 for arm, _, _ in ARMS for left, right in (('repair', 'adaptive'), ('repair', 'static'), ('reverse', 'repair'))}
    activity = {}
    for arm, _, _ in ARMS:
        for mode in ('repair', 'reverse'):
            rows = [x for x in diagnostics if x[0] == arm and x[1] == mode]
            activity[f'{arm}_{mode}'] = {'sessions': len(rows), 'attempts': sum(x[2] for x in rows),
                                       'updates': sum(x[3] for x in rows), 'skips': sum(x[4] for x in rows),
                                       'selected_total': sum(x[5] for x in rows),
                                       'zero_update_sessions': sum(x[3] == 0 for x in rows)}
    report = {'official_equal_cell_scheme_equal': {f'{arm}_{mode}': float(np.mean([
                  np.mean([official[arm, mode, o] for o in ORDERS if o.startswith(scheme)])
                  for scheme in ('within', 'full', 'runs')])) for arm, _, _ in ARMS for mode in MODES},
              'people_order_intervals': contrasts, 'activity': activity, 'legacy_near_ties': ties,
              'output_sha256': hashes, 'original_output_sha256': original_hashes,
              'manifest_sha256': sha(manifest), 'library_sha256': sha(library), 'extension_sha256': sha(extension),
              'seed_aggregation': 'score each seed independently, then average metrics',
              'population': 'held-out people, new-person path only; no routed package result'}
    target = ROOT / 'evidence/next_wave_16nov/lane5_repair_results.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True); f.write('\n')
    print(f'validated complete matrix and saved {target}', flush=True)


if __name__ == '__main__':
    main()
