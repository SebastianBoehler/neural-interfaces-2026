"""Score §306 whole decoders only after the complete source-checked matrix exists."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guards
import inputs
from score_s_v66p import person_accuracy, summarize
from style import sha

sys.path.insert(0, str(inputs.PROD / 'experiments'))
import protocol

PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))


def check_completed(job):
    rows = subprocess.check_output(['sacct', '-n', '-X', '-P', '-j', job, '-o', 'JobID,State,ExitCode'], text=True).splitlines()
    records = [x.split('|') for x in rows]
    if ({r[0] for r in records} != {f'{job}_{i}' for i in range(10)} or
            any(len(r) != 3 or r[1:] != ['COMPLETED', '0:0'] for r in records)):
        raise ValueError('production array is not the complete successful 10-fold matrix')


def validate(root, subject, session, perms):
    folder = root / 'runs/expert306_full'
    names = {f'{p}.npz' for p in PEOPLE}
    actual = {p.name for p in folder.glob('*.npz')}
    if actual != names: raise ValueError(f'incomplete fold matrix missing={sorted(names-actual)} extra={sorted(actual-names)}')
    lock = root / 'runs/input306/inputs.json'; lock_sha = sha(lock)
    if lock_sha != lock.with_suffix('.sha256').read_text().split()[0]: raise ValueError('input lock mismatch')
    if json.loads((root / 'runs/expert306_smoke/passed.json').read_text())['input_lock_sha256'] != lock_sha:
        raise ValueError('smoke used different inputs')
    locked = json.loads(lock.read_text())['files']
    if set(locked) != set(map(str, inputs.paths())): raise ValueError('input set mismatch')
    for path, digest in locked.items():
        if sha(path) != digest: raise ValueError(f'changed input {path}')
    inputs.check_sources()
    merged = {(order, mode): np.full((len(subject), 3), np.nan) for order in guards.ORDERS for mode in guards.MODES}
    hashes = {}
    for person in PEOPLE:
        path = folder / f'{person}.npz'; digest = sha(path)
        meta, results = guards.load_fold(path, person, subject, session, perms, lock_sha, 'full')
        for (order, mode), (rows, p) in results.items():
            target = merged[order, mode]
            if np.isfinite(target[rows]).any(): raise ValueError('duplicate fold coverage')
            target[rows] = p
        if sha(path) != digest: raise ValueError('fold changed while reading')
        hashes[path.name] = digest
    if any(not np.isfinite(x).all() for x in merged.values()): raise ValueError('missing rows')
    return merged, hashes, lock_sha


def main():
    root = Path(os.environ['EXPERT_ROOT']).resolve(strict=True)
    job = os.environ['EXPERT_JOB_ID']
    if not job.isdigit(): raise ValueError('invalid production job id')
    output = root / 'runs/expert306_results/results.json'
    if output.exists(): raise FileExistsError(output)
    check_completed(job)
    data = inputs.PROD / 'data/proteus_windows.npz'
    with np.load(data, allow_pickle=False) as z: subject, session = z['subject'].copy(), z['session'].copy()
    with np.load(inputs.PROD / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in guards.ORDERS}
    if any(not np.array_equal(np.sort(p), np.arange(len(subject))) for p in perms.values()):
        raise ValueError('bad pinned order')
    candidates, hashes, lock_sha = validate(root, subject, session, perms)
    # Only now, after the full output and provenance matrix passed, open held labels/interfaces.
    with np.load(data, allow_pickle=False) as z:
        y, iface = z['y'].copy(), z['interface'].copy()
    stream = np.char.add(np.char.add(subject, '|'), session)
    cell = np.char.add(np.char.add(stream, '|'), iface)
    person = np.array([x.rsplit('/', 1)[-1] for x in subject])
    d = dict(subject=subject, session=session, interface=iface, stream=stream)
    official, persons = {}, {}
    for name in guards.MODES:
        official[name] = {}; persons[name] = {}
        for order in guards.ORDERS:
            p = candidates[order, name]
            official[name][order] = 100 * protocol.score(y, p.argmax(axis=1), d, np.ones(len(y), bool))['balanced_accuracy_cell_mean']
            persons[name][order] = person_accuracy(p, y, cell, person, PEOPLE)
    def equal_scheme(values):
        groups = ('within', 'full', 'runs')
        return float(np.mean([np.mean([values[o] for o in guards.ORDERS if o.startswith(g)]) for g in groups]))
    pairs = [('joint1', 'shared'), ('joint10', 'shared'), ('joint10', 'joint1'),
             ('joint1', 'joint1_equal'), ('joint10', 'joint10_equal'),
             ('joint1', 'joint1_prior'), ('joint10', 'joint10_prior')]
    differences = {}
    for a, b in pairs:
        delta = np.stack([persons[a][o] - persons[b][o] for o in guards.ORDERS])
        summary = summarize(delta, guards.ORDERS, PEOPLE)
        summary['official_equal_cell_points'] = equal_scheme({o: official[a][o]-official[b][o] for o in guards.ORDERS})
        differences[f'{a}-{b}'] = summary
    summary = dict(registration='§306', production_job=job, input_lock_sha256=lock_sha, folds=hashes,
                   official_equal_cell_by_order=official,
                   official_equal_cell_scheme_equal={a: equal_scheme(v) for a, v in official.items()},
                   differences=differences, whole_alternatives=True, chain_or_package_gain_claimed=False,
                   interpretation='new-person only; repeatedly selected development people')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(summary, f, sort_keys=True, indent=2); f.write('\n')
    print('scored complete 10-person/eight-order whole decoders; result sha256', sha(output), flush=True)


if __name__ == '__main__': main()
