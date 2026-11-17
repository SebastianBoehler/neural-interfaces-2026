"""Enumerate source pair choices while holding the six-person anchor fixed."""
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ORIGINAL = Path('/mnt/lustre/work/gehler/gfh527/ni26_complement372_v1_20261117')
def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if checksum(ORIGINAL/'source.sha256') != 'ec76c2a91b24fdd4d26c42b28d0a75403619d4adcd87241538b4f8b7b9bad0a1':
    raise ValueError('changed original producer manifest')
for line in (ORIGINAL/'source.sha256').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    if checksum(ORIGINAL/name) != expected:
        raise ValueError('changed original producer')
sys.path.insert(0, str(ORIGINAL))
import pilot372 as source

COHORT = ('P01','P02','P03','P09')
PAIRS = tuple(itertools.combinations(COHORT, 2))
ARMS = source.ARMS + ('shuffled',)


def inputs():
    proof = source.verify()
    streams, rows, meta, people = source.banks(proof)
    base = {p:np.stack([streams[p][o][0][np.argsort(streams[p][o][2])] for o in source.ORDERS]) for p in COHORT}
    return rows, meta, people, base


def original():
    source.pin(ORIGINAL/'before_labels.json', 'd184e7f311268171f2efc257ca03c1586549d43ba54773aabd5de118864569d2')
    card = json.loads((ORIGINAL/'before_labels.json').read_text())
    source.pin(ORIGINAL/'members.npz', card['archive_sha'])
    source.require(card['train'] == list(source.TRAIN) and card['held_labels_open'] is False, 'original scope')
    with np.load(ORIGINAL/'members.npz') as archive:
        saved = {key:archive[key] for key in archive.files}
    return card, saved


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    train = PAIRS[idx]
    test = tuple(p for p in COHORT if p not in train)
    root = Path('fold'+str(idx))
    root.mkdir()
    rows, meta, people, base = inputs()
    fitting = np.concatenate([rows[p] for p in train])
    y = source.labels(fitting, len(people))
    cell = np.char.add(meta['stream'][fitting], meta['interface'][fitting])
    w = source.weights(np.concatenate([base[p] for p in train], axis=1), y, cell)
    shuffled = w[1].clone()
    rng = np.random.default_rng(372)
    for group in np.unique(cell):
        for label in range(3):
            r = np.flatnonzero((cell == group) & (y == label))
            shuffled[r] = w[1,rng.permutation(r)]
    source.require(torch.equal(w[1].sort().values, shuffled.sort().values)
                   and not torch.equal(w[1], shuffled), 'shuffle no-op')
    weights = torch.cat([w, shuffled[None]], dim=0)
    with np.load(source.DATA_ROOT/'data/proteus_windows.npz') as d:
        x = torch.from_numpy(d['X']).float()
    x /= x[fitting].square().mean().sqrt()
    target = torch.from_numpy(y).long()
    _, smoke = source.fit(x[fitting],target,shuffled,1)
    traces, output = {}, {}
    if train == source.TRAIN:
        card, output = original()
        traces.update(card['traces'])
        source.require(card['weight_shas'] == [source.digest(v.numpy()) for v in w], 'original inputs differ')
    for arm, weight in zip(ARMS, weights):
        if arm in traces:
            continue
        model, traces[arm] = source.fit(x[fitting], target, weight, 24)
        with torch.no_grad():
            for person in test:
                query = x[rows[person]]
                q = model(query).softmax(-1).numpy()
                source.require(np.allclose(q, model(query.flip(0)).softmax(-1).flip(0).numpy(), atol=1e-6)
                               and np.allclose(q[:1], model(query[:1]).softmax(-1).numpy(), atol=1e-6)
                               and np.array_equal(base[person] + 0.*q, base[person]), 'reset/order/zero')
                output[person+'_'+arm] = q
    source.require(len(output) == 8 and len({v['initial'] for v in traces.values()}) == 1
                   and len({v['batches'] for v in traces.values()}) == 1
                   and len({v['final'] for v in traces.values()}) == 4, 'matched fit activation')
    np.savez_compressed(root/'members.npz', **output)
    (root/'proof.json').write_text(json.dumps(dict(sha=checksum(root/'members.npz'), train=train, test=test,
        smoke=smoke, traces=traces, reused_three=train == source.TRAIN, scoring_labels_open=False), indent=2))


if __name__ == '__main__':
    main()
