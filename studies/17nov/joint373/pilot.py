"""Jointly fit a waveform representation and a causal unordered-session decoder."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
OWN = ROOT/'ni26_prefix359_v1_20261117'
BUILD = ROOT/'ni26_prefix358_v2_20261117'


def check_manifest(folder, expected):
    manifest = folder/'source.sha256'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected
    for row in manifest.read_text().splitlines():
        hexdigest, name = row.split(maxsplit=1)
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == hexdigest


check_manifest(OWN, 'b6777986c49bd9c38f5760d4f601c4b45739ef51eb4807afa4146b973b875325')
check_manifest(BUILD, 'a0a629f58f277a4f0681b69d062ee22752080296974954d47a6d1be95f32e1e0')
sys.path[:0] = [str(OWN), str(BUILD)]
from build import restore, verify as verify_body, FIT_PEOPLE
sys.path.insert(0, str(OWN))
from pilot import COHORT, METRIC_SHA, banks, verify
assert Path(sys.modules["pilot"].__file__).resolve() == OWN/"pilot.py"
from inputs import ORDERS, labels, digest, pin, sha, state_sha, permutations
from recipe336 import load, official
from report import score


class Joint(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        self.item = nn.Sequential(nn.LayerNorm(1152), nn.Linear(1152, 24), nn.Tanh())
        self.move = nn.Linear(48, 3)
        self.gate = nn.Linear(48, 1)

    def encode(self, x):
        features = []
        handle = self.body.final_layer.register_forward_pre_hook(lambda module, args: features.append(args[0]))
        try:
            logits = self.body(x)
        finally:
            handle.remove()
        assert len(features) == 1 and logits.shape == (len(x), 3)
        return logits, self.item(features[0])

    def predict(self, logits, items, prefix):
        previous = torch.cat((torch.zeros_like(items[:1]), items[:-1].cumsum(0)))
        count = torch.arange(len(items), dtype=items.dtype).clamp_min(1)[:, None]
        features = torch.cat((items, previous / count if prefix else items), dim=1)
        return logits + self.gate(features).sigmoid() * self.move(features).tanh()


def episodes(rows, data, y):
    rng = np.random.default_rng(373)
    streams = data['stream'][rows]
    sessions = np.unique(streams)
    cells = np.char.add(np.char.add(streams, '|'), data['interface'][rows])
    weights = np.zeros(len(rows))
    for session in sessions:
        active = streams == session
        for cell in np.unique(cells[active]):
            for task in range(3):
                m = (cells == cell) & (y == task)
                assert m.any()
                weights[m] = active.sum() / m.sum()
    weights /= weights.mean()
    selections = [rng.choice(np.flatnonzero(streams == rng.choice(sessions)), 32, replace=False)
                  for _ in range(64)]
    return selections, torch.tensor(weights, dtype=torch.float32)


def fit(initial, x, y, batches, weights, prefix):
    model = copy.deepcopy(initial).eval()
    optimizer = torch.optim.AdamW([
        dict(params=model.body.parameters(), lr=3e-5),
        dict(params=list(model.item.parameters()) + list(model.move.parameters()) + list(model.gate.parameters()),
             lr=5e-4)], weight_decay=.001)
    start = state_sha(model.body)
    gradients = []
    for batch in batches:
        optimizer.zero_grad()
        logits, items = model.encode(x[batch])
        pred = model.predict(logits, items, prefix)
        cost = (F.cross_entropy(pred, y[batch], reduction='none') * weights[batch]).mean()
        assert torch.isfinite(cost)
        cost.backward()
        if not gradients:
            for group in (model.body, model.item, model.move, model.gate):
                gradients.append(max(p.grad.abs().max().item() for p in group.parameters() if p.grad is not None))
            assert min(gradients) > 0
        optimizer.step()
    assert state_sha(model.body) != start
    return model, dict(initial_body=start, final_body=state_sha(model.body), gradients=gradients,
                       batches=digest(np.stack(batches)), steps=len(batches))


@torch.no_grad()
def guards(model, x):
    logits, items = model.encode(x)
    first = model.predict(logits, items, True)
    # Fresh single call with precisely the same previous set in reverse order.
    rev = [*range(len(x)-2, -1, -1), len(x)-1]
    shuffled = model.predict(logits[rev], items[rev], True)
    torch.testing.assert_close(first[-1], shuffled[-1], atol=3e-6, rtol=0)
    torch.testing.assert_close(first[:6], model.predict(logits[:6], items[:6], True), atol=3e-6, rtol=0)
    assert (first[-1] - model.predict(logits[-1:], items[-1:], True)[0]).abs().max() > 1e-6
    zero = copy.deepcopy(model)
    zero.move.weight.zero_()
    zero.move.bias.zero_()
    assert torch.equal(zero.predict(logits, items, True), logits)
    # Single-window exact inference within FP32 tolerance.
    one = torch.cat([model.encode(xx[None])[1] for xx in x[:6]])
    torch.testing.assert_close(one, items[:6], atol=2e-4, rtol=1e-4)


def main():
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.manual_seed(373)
    proof = verify()
    streams, held, meta, people = banks(proof)
    data, _, _, people2 = load()
    assert 'y' not in data and np.array_equal(people2, people)
    rows = np.flatnonzero(np.isin(people, FIT_PEOPLE))
    assert set(people[rows]) == set(FIT_PEOPLE) and set(people)-set(FIT_PEOPLE) == set(COHORT)
    runtime = restore(data, rows, verify_body())
    runtime.set_k(None)
    initial = Joint(runtime.net).eval()
    x = torch.from_numpy(runtime._aligned(rows, True))
    y = labels(rows, len(people))
    batches, weights = episodes(rows, data, y)
    y = torch.from_numpy(y.astype(np.int64))
    guards(initial, x[:9])
    _, smoke = fit(initial, x, y, batches[:1], weights, True)
    print('source-only smoke passed', smoke, flush=True)
    traces, outputs = {}, {}
    perm = permutations(len(people))
    with np.load('controls.npz') as control:
        assert np.array_equal(control['subject'], meta['subject'])
        for person, r in held.items():
            for order in ORDERS:
                p, _, positions = streams[person][order]
                assert np.array_equal(r[positions], perm[order][np.isin(perm[order], r)])
                outputs[person, order, 'A6'] = p[np.argsort(positions)]
                for name in ('frozen', 'JS'):
                    outputs[person, order, name] = np.eye(3)[control[name+'_'+order][r]]
    for arm in ('current', 'prefix'):
        model, traces[arm] = fit(initial, x, y, batches, weights, arm == 'prefix')
        guards(model, x[:9])
        for person, r in held.items():
            assert not np.intersect1d(rows, r).size and len(set(meta['stream'][r])) == 1
            chunks = torch.from_numpy(runtime._aligned(r, True)).split(32)
            with torch.no_grad():
                values = [model.encode(chunk) for chunk in chunks]
                logits, items = [torch.cat(v) for v in zip(*values)]
                for order in ORDERS:
                    positions = streams[person][order][2]
                    p = model.predict(logits[positions], items[positions], arm == 'prefix').softmax(1).numpy()
                    assert np.isfinite(p).all()
                    outputs[person, order, arm] = p[np.argsort(positions)]
        print(arm, 'completed', flush=True)
    assert traces['current']['initial_body'] == traces['prefix']['initial_body']
    assert traces['current']['batches'] == traces['prefix']['batches']
    assert len(outputs) == 4*8*5
    np.savez_compressed('outputs.npz', **{'|'.join(k): v for k, v in outputs.items()})
    audit = dict(held=COHORT, source=FIT_PEOPLE, held_labels_opened=False, training_rows=digest(rows),
                 traces=traces, outputs_sha=sha(Path('outputs.npz')), controls_sha=sha(Path('controls.npz')),
                 before_labels=True, source_manifest=sha(Path('source.sha256')))
    Path('before_labels.json').write_text(json.dumps(audit, indent=2))
    metric = official()
    pin(Path(metric.__file__), METRIC_SHA)
    result = {}
    for ref in ('A6', 'frozen', 'JS'):
        aligned = {**outputs, **{(p, o, 'A6'): outputs[p, o, ref] for p in COHORT for o in ORDERS}}
        result[ref] = score(metric, labels, meta, people, COHORT, ORDERS, aligned)
    result['seconds'] = time.monotonic()-start
    Path('result.json').write_text(json.dumps(result, indent=2))
    print('complete', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
