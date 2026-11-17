"""Use the original §323 S0C body and predict every held row without its labels."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from relation import LIMIT, adjust, blocks, fit

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PRODUCER = WORK / 'ni26_s323_normsplit_20261117'
CHAIN = WORK / 'ni26_s323_chain_20261117'
SOURCE = '88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5'
PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bytes_sha(rows):
    return hashlib.sha256(rows.tobytes()).hexdigest()


def verified_code():
    manifest = PRODUCER / 'source323.sha256'
    if sha(manifest) != SOURCE:
        raise ValueError('frozen model source manifest changed')
    for line in manifest.read_text().splitlines():
        h, name = line.split(maxsplit=1)
        if sha(PRODUCER / name) != h:
            raise ValueError(f'frozen model source changed: {name}')
    sys.path[:0] = [str(PRODUCER), str(PRODUCER / 'experiments')]
    import data323
    import model322
    import modes
    return data323, model322, modes


def streamed(predict, model, raw, rows, streams, rank, scale):
    out = np.empty((len(rows), 3), dtype=np.float64)
    for session in dict.fromkeys(streams[rows]):
        positions = np.flatnonzero(streams[rows] == session)
        positions = positions[np.argsort(rank[rows[positions]], kind='stable')]
        out[positions] = predict(model, raw[rows[positions]] / scale, 'cpu')
    if not np.isfinite(out).all() or not np.allclose(out.sum(axis=1), 1, atol=1e-6):
        raise ValueError('invalid frozen source probabilities')
    return out


def frozen(held, seed, data, source, test, rank, declarations, model, modes):
    key = f'{held}_s{seed}'
    folder = PRODUCER / 'output' / key
    fold_file = folder / 'result.npz'
    if sha(fold_file) != declarations[key]:
        raise ValueError(f'archived fold changed: {key}')
    with np.load(fold_file, allow_pickle=False) as z:
        m = json.loads(str(z['metadata']))
        if (not np.array_equal(z['index'], test)
                or m['source_rows_sha256'] != bytes_sha(source)
                or m['held_rows_sha256'] != bytes_sha(test)
                or m['source_manifest_sha256'] != SOURCE or m['held_labels_opened']
                or {p.split('/')[-1] for p in m['fit_people']} != set(PEOPLE) - {held}
                or m['permutation_sha256'] != sha(PRODUCER / 'runs/shuffle_perms.npz')
                or (m['held'], m['seed']) != (held, seed)):
            raise ValueError(f'bad frozen fold/exposure: {key}')
        bank = {o: z[f'S0C|{o}'].copy() for o in ORDERS}
    checkpoint = folder / 'S0.pt'
    if sha(checkpoint) != m['checkpoints']['S0']:
        raise ValueError(f'frozen body changed: {key}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (payload['held'], payload['seed'], payload['norm'], payload['skip'], payload['scale']) != (
            held, seed, 'session', False, m['scale']):
        raise ValueError('wrong frozen body')
    model.SEED = seed
    body = model.Factor('session', False, 'cpu').eval()
    body.load_state_dict(payload['state'], strict=True)
    before = model.param_hash(body)
    scale = m['scale']
    raw = data['X']
    # Prefixes start at the actual session boundary in within0. No gold is needed.
    subset = np.concatenate([
        ids[np.argsort(rank[ids], kind='stable')[:8]]
        for session in dict.fromkeys(data['stream'][test])
        for ids in [test[data['stream'][test] == session]]
    ])
    reproduced = streamed(modes.probabilities, body, raw, subset, data['stream'], rank, scale)
    error = float(np.abs(reproduced - bank['within0'][np.searchsorted(test, subset)]).max())
    if error > 2e-5:
        raise ValueError(f'frozen S0C CPU/archive prefix parity failed: {error}')
    p = streamed(modes.probabilities, body, raw, source, data['stream'], rank, scale)
    if before != model.param_hash(body):
        raise ValueError('frozen body parameters changed')
    return p, bank, error, {str(checkpoint): sha(checkpoint), str(fold_file): declarations[key]}


def run(held):
    if held not in PEOPLE:
        raise ValueError('invalid holdout')
    start = time.monotonic()
    torch.set_num_threads(2)
    data323, model, modes = verified_code()
    data, source, test = data323.load(held)
    with np.load(PRODUCER / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perm = z['within|0|all_loso']
    if not np.array_equal(np.sort(perm), np.arange(len(data['X']))):
        raise ValueError('invalid archived permutation')
    rank = np.empty(len(perm), dtype=np.int64)
    rank[perm] = np.arange(len(perm))
    declaration = PRODUCER / 'runs/n323_streams/validation.json'
    declarations = json.loads(declaration.read_text())
    if declarations['source_manifest'] != SOURCE or set(declarations['folds']) != {
            f'{p}_s{s}' for p in PEOPLE for s in (0, 1)}:
        raise ValueError('incomplete frozen producer')
    first, second = [frozen(held, s, data, source, test, rank, declarations['folds'], model, modes)
                     for s in (0, 1)]
    offset = np.mean([first[0], second[0]], axis=0)
    f = blocks(data['X'])
    arrays = {}
    stats = {}
    for arm, features in zip(('instant', 'lag'), f, strict=True):
        mu, sd, weight, iterations = fit(features[source], data['y'][source], offset)
        effect = LIMIT * np.tanh((features[test] - mu) / sd @ weight / LIMIT)
        if np.count_nonzero(effect) != effect.size:
            raise ValueError('inactive EEG residual')
        stats[arm] = dict(fit_iterations=iterations, residual_max=float(abs(effect).max()),
                          params=weight.size, parameters_sha256=bytes_sha(weight))
        for o in ORDERS:
            base = np.mean([first[1][o], second[1][o]], axis=0)
            arrays[f'frozen|{o}'] = base
            if not np.array_equal(adjust(base, np.zeros_like(base)), base):
                raise ValueError('residual-off changed the frozen posterior')
            arrays[f'{arm}|{o}'] = adjust(base, effect)
    if not all(np.max(abs(arrays[f'instant|{o}'] - arrays[f'lag|{o}'])) > 1e-6 for o in ORDERS):
        raise ValueError('lag perturbation was inactive')
    manifest = ROOT / 'source.sha256'
    meta = dict(held=held, fit_people=sorted(set(PEOPLE) - {held}), fit_rows=len(source),
                held_rows=len(test), held_labels_opened=False,
                source_rows_sha256=bytes_sha(source), held_rows_sha256=bytes_sha(test),
                device='CPU', parity_max=max(first[2], second[2]), mechanism=stats,
                sources=first[3] | second[3] | {str(manifest): sha(manifest),
                    str(declaration): sha(declaration), str(data323.DATA): data323.DATA_SHA,
                    str(PRODUCER / 'source323.sha256'): SOURCE},
                elapsed_seconds=time.monotonic() - start)
    with (ROOT / f'{held}.npz').open('xb') as stream:
        np.savez_compressed(stream, index=test, metadata=json.dumps(meta), **arrays)
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
