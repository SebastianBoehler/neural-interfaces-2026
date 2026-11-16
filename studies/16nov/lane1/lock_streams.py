"""Lock all staged lane1 streams for the existing guarded scorer."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
FAMILIES = ('eegn', 'eegn_dist2', 'fb_coral3', 'fb_coral10')


def merge_entries(groups):
    result = {}
    for group in groups:
        for name, digest in group.items():
            if not name.endswith('.npz'):
                continue
            if name in result:
                raise ValueError(f'duplicate stream: {name}')
            result[name] = digest
    return result


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    directory = ROOT / 'runs/lane1_inputs'
    groups = [json.loads((directory / f'{name}.json').read_text()) for name in ('coral', 'plain', 'dist')]
    if [g.get('group') for g in groups] != ['coral', 'plain', 'dist']:
        raise ValueError('staging group mismatch')
    records = merge_entries([g['inputs_sha256'] for g in groups])
    expected = {f'ttt18{o}_{f}_base-all_loso-s{s}.npz' for o in ORDERS for f in FAMILIES for s in range(6)}
    if set(records) != expected:
        raise ValueError('need the exact 192 distinct candidate streams')
    for name, digest in records.items():
        if sha(ROOT / 'runs/diag/ttt/probs18' / name) != digest:
            raise ValueError(f'staged stream changed: {name}')
    path, lock = directory / 'streams.sha256', directory / 'lock.json'
    if path.exists() or lock.exists():
        raise FileExistsError('stream lock already exists')
    with path.open('x') as f:
        for name, digest in sorted(records.items()):
            f.write(f'{digest}  {name}\n')
    with lock.open('x') as f:
        json.dump({'manifest_sha256': sha(path), 'groups_sha256': {
            g: sha(directory / f'{g}.json') for g in ('coral', 'plain', 'dist')}}, f, sort_keys=True, indent=2)
    print(f'locked 192 streams: manifest_sha256={sha(path)}', flush=True)


if __name__ == '__main__':
    main()
