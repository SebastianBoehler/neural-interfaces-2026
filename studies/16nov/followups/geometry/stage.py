"""Copy only fully checked §295 deltas into the private chain workspace."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('NI26_WS', Path(__file__).resolve().parents[4])).resolve()
sys.path.insert(0, str(ROOT / 'studies/16nov/followups/geometry'))
from chain import ARMS, ORDERS, sha, validate_delta  # noqa: E402


def expected_names(include_zero):
    return {f'{a}_{o}.npz' for a in ARMS if include_zero or a != 'geo_zero' for o in ORDERS}


def main():
    src = Path(os.environ['S295_PRODUCER_ROOT']).resolve(strict=True)
    expected_source = os.environ['S295_PRODUCER_SHA']
    if sha(src / 'source.sha256') != expected_source:
        raise ValueError('producer source lock mismatch')
    source = src / 'runs/s295_geometry/deltas'
    target = ROOT / 'runs/s295_geometry/deltas'
    treatment = source.parent / 'treatment_manifest.json'
    control = target.parent / 'zero_manifest.json'
    output = target.parent / 'manifest.json'
    if output.exists():
        raise FileExistsError(output)
    tm = json.loads(treatment.read_text()); zm = json.loads(control.read_text())
    if (set(tm['files']) != expected_names(False) or set(zm['files']) != expected_names(True) - expected_names(False) or
            tm['source_manifest_sha256'] != expected_source or len(tm['shards']) != 10):
        raise ValueError('incomplete geometry manifests')
    for name, digest in tm['shards'].items():
        if sha(src / 'runs/s295_geometry/shards' / name) != digest:
            raise ValueError(f'fold shard changed: {name}')
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        n = len(z['subject'])
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        order_hash = {o: hashlib.sha256(z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64).tobytes()).hexdigest() for o in ORDERS}
    files = {}
    for name, expected in sorted({**zm['files'], **tm['files']}.items()):
        arm, order = name[:-4].rsplit('_', 1)
        origin = target / name if arm == 'geo_zero' else source / name
        destination = target / name
        if not origin.is_file() or sha(origin) != expected:
            raise ValueError(f'missing or changed geometry delta: {name}')
        if arm != 'geo_zero' and destination.exists():
            raise FileExistsError(destination)
        with np.load(origin, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata'].item()))
            if (metadata.get('arm'), metadata.get('order'), metadata.get('order_sha256')) != (arm, order, order_hash[order]):
                raise ValueError(f'delta identity or order mismatch: {name}')
            validate_delta(z['delta'], n, zero=arm == 'geo_zero')
        if arm != 'geo_zero':
            if metadata.get('source_manifest_sha256') != expected_source:
                raise ValueError(f'producer source mismatch: {name}')
            with origin.open('rb') as f, destination.open('xb') as g:
                shutil.copyfileobj(f, g)
            if sha(origin) != expected or sha(destination) != expected:
                raise ValueError(f'geometry copy differs: {name}')
        files[name] = expected
    with output.open('x') as f:
        json.dump({'files': files, 'producer_source_sha256': expected_source,
                   'treatment_manifest_sha256': sha(treatment), 'shards': tm['shards']}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'staged 40 checked deltas; manifest {output}: {sha(output)}', flush=True)


if __name__ == '__main__':
    main()
