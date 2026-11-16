"""§312 complete source-exclusive producer; writes probabilities without scoring held labels."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ['NI26_WS'])
sys.path.insert(0, str(ROOT / 'experiments'))
import protocol
from data import sha
from encoder import fingerprint
from training import fit, probabilities, sources

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
ARMS = ('S', 'F', 'T', 'C')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('task', type=int)
    args = parser.parse_args()
    if not 0 <= args.task < 160:
        raise ValueError('expected task0..159')
    held = f'P{args.task % 10 + 1:02d}'
    seed = args.task // 10 % 2
    arch = ('fb', 'sh')[args.task // 20 % 2]
    arm = ARMS[args.task // 40]
    if not torch.cuda.is_available():
        raise ValueError('explicit CUDA route unavailable')
    torch.set_num_threads(6)
    cache, output = Path(os.environ['NI26_CACHE']), Path(os.environ['NI26_OUT'])
    output.mkdir(exist_ok=True)
    destination = output / f'{arch}_{arm}_s{seed}_{held}.npz'
    if destination.exists():
        raise FileExistsError(destination)
    d = protocol.load()
    target, fit_people, val = sources(d, held, seed)
    inner = [p for p in fit_people if p != val]
    chosen, _, selected_meta = fit(d, cache, inner, arch, arm, seed, 'cuda', val_people=[val])
    epochs = selected_meta['supervised']['selected_epochs']
    del chosen
    net, scale, meta = fit(d, cache, fit_people, arch, arm, seed, 'cuda', epochs=epochs)
    rows = np.flatnonzero(d['subject'] == target)
    before = fingerprint(net)
    p = probabilities(net, d['X'][rows], scale, 'cuda')
    n = min(17, len(rows))
    if not np.allclose(p[:n], probabilities(net, d['X'][rows[:n]], scale, 'cuda'), atol=2e-5, rtol=2e-5):
        raise ValueError('prefix/batch-shape probability guard failed')
    # Interleaving another source person's inference cannot mutate a static model.
    source_rows = np.flatnonzero(d['subject'] == fit_people[0])[:10]
    probabilities(net, d['X'][source_rows], scale, 'cuda')
    if fingerprint(net) != before:
        raise ValueError('inference changed model state')
    arrays = {}
    with np.load(ROOT / 'runs/shuffle_perms.npz') as permutations:
        for order in ORDERS:
            perm = permutations[f'{order[:-1]}|{order[-1]}|all_loso']
            if not np.array_equal(np.sort(perm), np.arange(len(d['X']))):
                raise ValueError('invalid pinned full permutation')
            rank = np.empty(len(perm), dtype=int); rank[perm] = np.arange(len(perm))
            idx = np.argsort(rank[rows], kind='stable')
            q = probabilities(net, d['X'][rows[idx]], scale, 'cuda')
            restored = np.empty_like(q); restored[idx] = q
            if not np.allclose(restored, p, atol=2e-5, rtol=2e-5):
                raise ValueError('static order invariance failed')
            arrays[f'proba_{order}'] = restored
    checkpoint = output / f'{arch}_{arm}_s{seed}_{held}.pt'
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save(dict(state=net.state_dict(), scale=scale, architecture=arch, metadata=meta), checkpoint)
    meta.update(arm=arm, architecture=arch, seed=seed, held=held, validation_person=val,
        selected_epochs=epochs, inner_fit_people=inner, selected=selected_meta,
        orders=ORDERS, source_manifest=sha(ROOT / 'source.sha256'),
        data_sha256=sha(ROOT / 'data/proteus_windows.npz'),
        permutation_sha256=sha(ROOT / 'runs/shuffle_perms.npz'), checkpoint_sha256=sha(checkpoint),
        model_sha256=before, state_policy='stateless;ordinaryBN frozen at inference')
    if target in meta['fit_people'] or target in inner or val in inner:
        raise ValueError('source exclusion failed')
    with destination.open('xb') as out:
        np.savez(out, index=rows, metadata=json.dumps(meta), **arrays)
    print(json.dumps(dict(output=str(destination), metadata=meta)), flush=True)


if __name__ == '__main__':
    main()
