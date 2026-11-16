"""Validate and copy completed candidate streams into the private chain workspace."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
GROUPS = {'coral': ('fb_coral3', 'fb_coral10'), 'eeg': ('eegn', 'eegn_dist2')}


def check_output(index, proba, n):
    if (not np.array_equal(index, np.arange(n)) or proba.shape != (n, 3) or
            not np.isfinite(proba).all() or (proba < 0).any() or
            not np.allclose(proba.sum(axis=1), 1, atol=1e-6)):
        raise ValueError('candidate indices or probabilities are invalid')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(group):
    if group not in GROUPS:
        raise ValueError(f'unknown group: {group}')
    source = ROOT / 'train/runs/diag/ttt/probs18'
    target = ROOT / 'chain/runs/diag/ttt/probs18'
    outdir = ROOT / 'chain/runs/lane1_inputs'; outdir.mkdir(exist_ok=True)
    manifest = outdir / f'{group}.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    inputs = {}
    for family in GROUPS[group]:
        for seed in range(6):
            side = source / f'{family}_s{seed}_fit.json'
            info = json.loads(side.read_text())
            if (info.get('family') != family or info.get('seed') != seed or info.get('smoke') or
                tuple(info.get('orders', ())) != ORDERS or
                set(info.get('fold_fingerprints', {})) != {f'P{i:02d}' for i in range(1, 11)}):
                raise ValueError(f'invalid fit provenance {side}')
            inputs[side.name] = sha(side)
            for order in ORDERS:
                name = f'ttt18{order}_{family}_base-all_loso-s{seed}.npz'
                src, dst = source / name, target / name
                if not src.is_file() or src.is_symlink() or dst.exists() or dst.is_symlink():
                    raise ValueError(f'missing source, linked source, or existing destination: {name}')
                with np.load(src, allow_pickle=False) as z:
                    check_output(z['index'], z['proba'], 2655)
                digest = sha(src)
                with src.open('rb') as f, dst.open('xb') as g:
                    shutil.copyfileobj(f, g)
                if sha(dst) != digest or sha(src) != digest:
                    raise ValueError(f'copy or source changed: {name}')
                inputs[name] = digest
    with manifest.open('x') as f:
        json.dump({'group': group, 'inputs_sha256': inputs}, f, sort_keys=True, indent=2)
    print(f'{group}: validated 96 streams and 12 fit manifests; copied privately', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
