"""Publish 192 full-recording streams only after the complete §304 fold matrix validates."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'experiments')]
import protocol
import validate as V


def main():
    out = ROOT / 'runs/coral304_streams'
    if out.exists():
        raise FileExistsError(out)
    d, folds, perms = V.inputs(ROOT, protocol)
    expected = {f'{a}_s{s}_{h}.npz' for a in V.ARMS for s in range(6) for h in folds}
    actual = {p.name for p in (ROOT / 'runs/coral304').glob('*.npz')}
    if actual != expected:
        raise ValueError(f'incomplete or unexpected fold files: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    arrays = {(a, s, o): np.full((2655, 3), np.nan) for a in V.ARMS for s in range(6) for o in V.ORDERS}
    hashes = {}
    for arm in V.ARMS:
        for seed in range(6):
            for held in sorted(folds):
                name = f'{arm}_s{seed}_{held}.npz'
                hashes[name], values = V.fold(ROOT, d, folds, perms, arm, seed, held)
                for order, (rows, probs) in values.items():
                    target = arrays[arm, seed, order]
                    if np.isfinite(target[rows]).any():
                        raise ValueError(f'duplicate rows {name}/{order}')
                    target[rows] = probs
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('missing global output rows')
    out.mkdir(parents=True)
    streams = {}
    for arm, family in zip(V.ARMS, V.FAMILIES):
        for seed in range(6):
            for order in V.ORDERS:
                name = f'ttt18{order}_{family}_base-all_loso-s{seed}.npz'
                metadata = json.dumps(dict(arm=arm, family=family, seed=seed, order=order,
                                           producer_manifest_sha256=V.MANIFEST), sort_keys=True)
                with (out / name).open('xb') as f:
                    np.savez_compressed(f, index=np.arange(2655, dtype=np.int64),
                                        proba=arrays[arm, seed, order], metadata=metadata)
                streams[name] = V.sha(out / name)
    if len(streams) != 192:
        raise ValueError('wrong stream count')
    with (out / 'streams.sha256').open('x') as f:
        for name, digest in sorted(streams.items()):
            f.write(f'{digest}  {name}\n')
    evidence = dict(producer_manifest_sha256=V.MANIFEST, data_sha256=V.DATA,
                    permutation_sha256=V.PERMS, producer_hashes=hashes, stream_hashes=streams,
                    counts=dict(folds=len(hashes), streams=len(streams)), held_accuracy_computed=False)
    with (out / 'validation.json').open('x') as f:
        json.dump(evidence, f, indent=2, sort_keys=True); f.write('\n')
    print(f'validated 240 folds and assembled 192 streams; no accuracy; manifest {V.sha(out / "streams.sha256")}')


if __name__ == '__main__':
    main()
