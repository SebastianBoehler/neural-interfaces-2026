"""Score only the complete, zero-checked §295 matrix via the existing scorer."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('NI26_WS', Path(__file__).resolve().parents[4])).resolve()
sys.path[:0] = [str(ROOT / 'studies/16nov/followups/geometry'), str(ROOT / 'experiments')]
from chain import ARMS, ORDERS, sha  # noqa: E402
import protocol  # noqa: E402

PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))
ALL = ('control', *ARMS)


def patch_scorer(source):
    old = "ARMS = ('control', 'zero_coral', 'S1', 'S2', 'S3', 'S4', 'S5')"
    if source.count(old) != 1:
        raise ValueError('expected one §281 scorer arm declaration')
    return source.replace(old, 'ARMS = ' + repr(ALL), 1).replace("'zero_coral'", "'geo_zero'")


def scorer():
    path = ROOT / 'studies/14nov/newmember/score_s_v66p.py'
    sys.path.insert(0, str(path.parent))
    namespace = {'__name__': 's295_existing_score', '__file__': str(path)}
    exec(compile(patch_scorer(path.read_text()), str(path), 'exec'), namespace)
    namespace['RECIPES'] = dict(namespace['RECIPES'])
    namespace['RECIPES'].update({a: ('none', 'none', a == 'geo_zero') for a in ARMS})
    return namespace


def verify_geometry(lock, manifest_path):
    names = {f'{a}_{o}.npz' for a in ARMS for o in ORDERS}
    if set(lock.get('files', {})) != names:
        raise ValueError('geometry manifest lacks the exact 40-file matrix')
    for arm in ARMS:
        for order in ORDERS:
            name = f'{arm}_{order}.npz'
            delta = ROOT / 'runs/s295_geometry/deltas' / name
            if not delta.is_file() or sha(delta) != lock['files'][name]:
                raise ValueError(f'{name}: delta hash mismatch')
            prov = ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
            m = json.loads(prov.read_text())
            entries = m['inputs_sha256']
            match = [v for k, v in entries.items() if Path(k).name == name]
            recorded_manifest = manifest_path
            if arm == 'geo_zero' and order == 'within0':
                # The independently completed zero smoke predates the full manifest.
                recorded_manifest = ROOT / 'runs/s295_geometry/zero_manifest.json'
                zero_lock = json.loads(recorded_manifest.read_text())
                if zero_lock['files'][name] != lock['files'][name]:
                    raise ValueError('within0 zero smoke manifest disagrees with full manifest')
            manifests = [v for k, v in entries.items() if Path(k).resolve() == recorded_manifest]
            if match != [lock['files'][name]] or manifests != [sha(recorded_manifest)]:
                raise ValueError(f'{name}: chain did not use the locked delta/manifest')


def main():
    manifest = Path(os.environ['S295_MANIFEST']).resolve(strict=True)
    if sha(manifest) != os.environ['S295_MANIFEST_SHA']:
        raise ValueError('geometry manifest lock mismatch')
    lock = json.loads(manifest.read_text())
    verify_geometry(lock, manifest)
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        subject, session, interface = (z[k].copy() for k in ('subject', 'session', 'interface'))
    d = {'subject': subject, 'session': session, 'interface': interface,
         'stream': np.char.add(np.char.add(subject, '|'), session)}
    s = scorer()
    # The existing collector enforces route trace, provenance and exact zero CP equality.
    outputs = s['collect'](ROOT, len(subject), len(np.unique(d['stream'])))
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        y = z['y'].copy()
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        if not np.array_equal(z['y|all_loso'], y):
            raise ValueError('official and replay labels differ')
        cell, person = z['cell|all_loso'], z['person|all_loso']
    if any(set(y[cell == c]) != {0, 1, 2} for c in np.unique(cell)):
        raise ValueError('cell lacks a class for official/helper metric parity')
    metric, official = {}, {}
    mask = np.ones(len(y), dtype=bool)
    for arm in ('control', *ARMS[1:]):
        for order in ORDERS:
            preds = [outputs[arm, order, b] for b in s['BASES']]
            metric[arm, order] = np.mean([s['person_accuracy'](x, y, cell, person, PEOPLE) for x in preds], axis=0)
            official[arm, order] = float(np.mean([100 * protocol.score(y, x.argmax(1), d, mask)['balanced_accuracy_cell_mean'] for x in preds]))
    def contrast(a, b):
        return s['summarize'](np.stack([metric[a, o] - metric[b, o] for o in ORDERS]), ORDERS, PEOPLE)
    pairs = [(a, 'control') for a in ARMS[1:]] + [('geo_d50', 'geo_s50'), ('geo_d25', 'geo_d50'), ('geo_d100', 'geo_d50')]
    report = {'official_equal_cell_by_order': {a: {o: official[a, o] for o in ORDERS} for a in ('control', *ARMS[1:])},
              'official_equal_cell_scheme_equal': {a: float(np.mean([np.mean([official[a, o] for o in ORDERS if o.startswith(k)])
                  for k in ('within', 'full', 'runs')])) for a in ('control', *ARMS[1:])},
              'absolute_people_order': {a: s['summarize'](np.stack([metric[a, o] for o in ORDERS]), ORDERS, PEOPLE)
                                        for a in ('control', *ARMS[1:])},
              'contrasts_people_order': {f'{a}_minus_{b}': contrast(a, b) for a, b in pairs},
              'manifest_sha256': sha(manifest), 'source_manifest_sha256': lock['producer_source_sha256'],
              'aggregation': 'mean of six separately argmax-scored base outputs',
              'interpretation': 'outer reconstructed-core posterior delta, fixed PC128/shadow/FT; new-person routing emulation, not packaged-core or returning-tail validation'}
    output = ROOT / 'evidence/next_wave_16nov/followups_geometry_results.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True); f.write('\n')
    print(f'complete guarded matrix scored: {output}')


if __name__ == '__main__':
    main()
