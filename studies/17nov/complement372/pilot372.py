"""Source-held complementarity diagnostic on a verified, six-person base producer."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path('/mnt/lustre/work/gehler/gfh527')
PARENT = BASE / 'ni26_prefix359_v1_20261117'
if hashlib.sha256((PARENT/'source.sha256').read_bytes()).hexdigest() != 'b6777986c49bd9c38f5760d4f601c4b45739ef51eb4807afa4146b973b875325':
    raise ValueError('changed §359 source manifest')
for line in (PARENT/'source.sha256').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    if hashlib.sha256((PARENT/name).read_bytes()).hexdigest() != expected:
        raise ValueError('changed §359 source file')
sys.path.insert(0, str(PARENT))
from pilot import DATA_ROOT, METRIC_SHA, ORDERS, banks, digest, labels, official, pin, require, sha, verify
from results372 import collect

TRAIN = ('P02', 'P03')
TEST = ('P01', 'P09')
ARMS = ('ordinary', 'error', 'confidence')


class Member(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv1d(41, 32, 1, bias=False)
        self.temporal = nn.Conv1d(32, 32, 31, groups=32, bias=False)
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        filtered = self.temporal(self.spatial(x))
        return self.head(filtered.square().mean(-1).clamp_min(1e-8).log())


def weights(base, y, cell):
    err = 1 + 2 * (base.argmax(-1) != y).mean(0)
    uncertainty = 1 - base.max(-1).mean(0)
    result = np.zeros((3, len(y)))
    for group in np.unique(cell):
        for label in range(3):
            idx = np.flatnonzero((cell == group) & (y == label))
            require(len(idx) > 0, 'empty source cell/class')
            uncertain = idx[np.argsort(uncertainty[idx], kind='stable')]
            result[0, idx] = 1 / len(idx)
            result[1, idx] = err[idx] / err[idx].sum()
            result[2, uncertain] = np.sort(result[1, idx])
    result *= len(y) / result[0].sum()
    require(np.allclose(result.sum(1), len(y)) and np.array_equal(np.sort(result[1]), np.sort(result[2]))
            and not np.allclose(result[1], result[2]) and not np.allclose(result[0], result[1]),
            'weight mass/control mismatch or inactive weighting')
    return torch.tensor(result, dtype=torch.float32)


def fit(x, y, w, epochs):
    torch.manual_seed(372)
    model = Member()
    rng = np.random.default_rng(372)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    initial = torch.cat([p.detach().flatten() for p in model.parameters()])
    batches = []
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(y)), (len(y)+31)//32):
            batches.append(idx)
            logits = model(x[idx])
            loss = (F.cross_entropy(logits, y[idx], reduction='none') * w[idx]).mean()
            require(torch.isfinite(loss), 'nonfinite weighted source loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    params = torch.cat([p.detach().flatten() for p in model.parameters()])
    require(torch.isfinite(params).all() and not torch.equal(initial, params), 'training no-op')
    return model.eval(), dict(steps=len(batches), batches=digest(np.concatenate(batches)),
                              initial=digest(initial.numpy()), final=digest(params.numpy()))


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    proof = verify()
    streams, rows, meta, people = banks(proof)
    require(set(proof['body_train_people']) == {'P04','P05','P06','P07','P08','P10'}
            and not set(TRAIN) & set(TEST), 'unexpected split')
    with np.load(DATA_ROOT/'data/proteus_windows.npz', allow_pickle=False) as data:
        x = torch.from_numpy(data['X']).float()
    require(x.shape == (len(people), 41, 480), 'unexpected EEG dimensions')
    train = np.concatenate([rows[p] for p in TRAIN])
    y = labels(train, len(x))
    cell = np.char.add(meta['stream'][train], meta['interface'][train])
    base = {p: np.stack([streams[p][o][0][np.argsort(streams[p][o][2])] for o in ORDERS]) for p in TRAIN+TEST}
    w = weights(np.concatenate([base[p] for p in TRAIN], axis=1), y, cell)
    x = x / x[train].square().mean().sqrt()
    target = torch.from_numpy(y).long()
    _, smoke = fit(x[train], target, w[1], 1)
    print('source-only smoke passed', smoke['steps'], flush=True)
    output, traces = {}, {}
    for arm, weight in zip(ARMS, w):
        model, traces[arm] = fit(x[train], target, weight, 24)
        with torch.no_grad():
            for person in TEST:
                subset = x[rows[person]]
                q = model(subset).softmax(-1).numpy()
                require(np.allclose(q[:1], model(subset[:1]).softmax(-1).numpy(), atol=1e-6)
                        and np.allclose(q, model(subset.flip(0)).softmax(-1).flip(0).numpy(), atol=1e-6),
                        'pointwise reset/order guard')
                z = base[person]
                require(np.array_equal(z*1. + q*0., z), 'zero-dose mismatch')
                output[person+'_'+arm] = q
        require(traces[arm]['initial'] == traces['ordinary']['initial'] and
                traces[arm]['batches'] == traces['ordinary']['batches'], 'matched fit differs')
    require(len({t['final'] for t in traces.values()}) == 3, 'weighted models identical')
    np.savez_compressed('members.npz', **output)
    before = dict(source_sha=sha(Path('source.sha256')), archive_sha=sha(Path('members.npz')),
                  train=TRAIN, test=TEST, body_sha=proof['body_sha256'], bank_sha=proof['banks_sha256'],
                  weight_shas=[digest(v.numpy()) for v in w], traces=traces, source_smoke=smoke,
                  held_labels_open=False)
    with open('before_labels.json', 'x') as f:
        json.dump(before, f, indent=2)
    scorer = official()
    pin(Path(scorer.__file__), METRIC_SHA)
    result = collect(scorer, labels, meta, people, rows, TEST, ORDERS, ARMS, output, base)
    result.update(before_labels_sha=sha(Path('before_labels.json')), wall_s=time.monotonic()-start)
    with open('result.json', 'x') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
