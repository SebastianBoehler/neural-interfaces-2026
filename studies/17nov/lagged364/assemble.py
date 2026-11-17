"""Check all ten label-free folds against the original frozen member before staging."""
import json
from pathlib import Path

import numpy as np

from chain import FAMILIES, HERE, OLD, ORDERS, ROOT, ROWS, name, original_streams, sha


def dump(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)


def assemble():
    old_streams = original_streams()
    out = ROOT / 'runs/representation_streams'
    with np.load(OLD / 'data/proteus_windows.npz', allow_pickle=False) as z:
        person = np.array([x.rsplit('/', 1)[-1] for x in z['subject']])
    if len(person) != ROWS:
        raise ValueError('wrong public dataset')
    people = set(person)
    scores = {a: {o: np.full((ROWS, 3), np.nan) for o in ORDERS}
              for a in ('frozen', *FAMILIES)}
    proof = dict(folds={}, inputs={}, streams={})
    for held in sorted(people):
        path = HERE / f'{held}.npz'
        with np.load(path, allow_pickle=False) as z:
            m = json.loads(str(z['metadata']))
            idx = np.flatnonzero(person == held)
            source = np.flatnonzero(person != held)
            import hashlib
            if not (np.array_equal(z['index'], idx) and m['held'] == held
                    and m['held_rows'] == len(idx) and m['fit_rows'] == len(source)
                    and m['source_rows_sha256'] == hashlib.sha256(source.tobytes()).hexdigest()
                    and m['held_rows_sha256'] == hashlib.sha256(idx.tobytes()).hexdigest()
                    and set(m['fit_people']) == people - {held}
                    and not m['held_labels_opened'] and m['parity_max'] <= 2e-5
                    and {a: p['params'] for a, p in m['mechanism'].items()} ==
                    {'instant': 1200, 'lag': 1200}):
                raise ValueError(f'{held}: wrong source exposure or mechanism')
            for p, h in m['sources'].items():
                if sha(Path(p)) != h:
                    raise ValueError(f'{held}: fold input changed: {p}')
            proof['inputs'].update(m['sources'])
            proof['inputs'][str(path)] = sha(path)
            proof['folds'][held] = {k: v for k, v in m.items() if k != 'sources'}
            for a, orders in scores.items():
                for o, v in orders.items():
                    v[idx] = z[f'{a}|{o}']
    for o in ORDERS:
        archive = name('n323s0c', o)
        original = OLD / 'runs/representation_streams' / archive
        if sha(original) != old_streams[archive] or sha(out / archive) != old_streams[archive]:
            raise ValueError('original frozen stream hash changed')
        with np.load(original, allow_pickle=False) as z:
            if not np.array_equal(z['proba'], scores['frozen'][o]):
                raise ValueError(f'{o}: folded frozen member differs bitwise from archive')
        proof['streams'][archive] = old_streams[archive]
        a, b = scores['instant'][o], scores['lag'][o]
        if np.max(abs(a - b)) <= 1e-6:
            raise ValueError(f'{o}: lag perturbation inactive')
        for arm, f in FAMILIES.items():
            p = out / name(f, o)
            v = scores[arm][o]
            if not (np.isfinite(v).all() and (v > 0).all() and np.allclose(v.sum(1), 1, atol=1e-12)):
                raise ValueError(f'{arm}/{o}: invalid held posterior')
            if np.max(abs(v - scores['frozen'][o])) <= 1e-6:
                raise ValueError(f'{arm}/{o}: matched residual inactive')
            meta = dict(family=f, order=o, seeds=[0, 1],
                        operation='bounded_residual_after_arithmetic_probability_mean')
            with p.open('xb') as stream:
                np.savez(stream, index=np.arange(ROWS), proba=v, metadata=json.dumps(meta))
            proof['streams'][p.name] = sha(p)
    for f in ('source.sha256', 'chain-source.sha256', 'runtime.sha256', 'finish-source.sha256'):
        q = HERE / f
        proof['inputs'][str(q)] = sha(q)
    dump(HERE / 'proof.json', proof)
    print('OK: 10 held folds, 8 archived frozen posteriors bitwise; 16 active matched streams; no gold',
          flush=True)


if __name__ == '__main__':
    assemble()
