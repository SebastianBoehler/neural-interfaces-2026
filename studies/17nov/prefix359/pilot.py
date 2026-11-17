"""Four-fold held-person pilot from fixed causal A6 and B1 support banks."""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

BASE = Path('/mnt/lustre/work/gehler/gfh527')
READY = BASE/'ni26_prefix357_v1_20261117'
BANK = BASE/'ni26_prefix358_v2_20261117'
sys.path.insert(0, str(READY))
from inputs import ORDERS, digest, labels, pin, require, sha
from probe import fit as original_fit, guards, stack
from readout import Readout
from recipe336 import BASE as DATA_ROOT, official
from report import score

COHORT = ('P01', 'P02', 'P03', 'P09')
BANK_PROOF = 'a16b5715ef7a53c0052f1d789f638d62a7cfff2924d04a21f252a3dac9fc0ad7'
READY_SOURCE = 'ff2087149a43aab176233b78a7325939d84b694befd488690ed85d5ae7197b06'
METRIC_SHA = '390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c'
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'


def verify():
    pin(READY/'source.sha256', READY_SOURCE)
    for line in (READY/'source.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        pin(READY/name, expected)
    pin(BANK/'proof.json', BANK_PROOF)
    proof = json.loads((BANK/'proof.json').read_text())
    require(proof['status'] == 'FOUR_BODY_EXCLUDED_PEOPLE_LABEL_FREE_BANKS_PASS'
            and proof['label_accessed'] is False and tuple(proof['cohort']) == COHORT
            and tuple(proof['orders']) == ORDERS and proof['optimizer_steps'] == 0, 'bank proof changed')
    pin(BANK/'banks.npz', proof['banks_sha256'])
    pin(DATA_ROOT/'data/proteus_windows.npz', DATA_SHA)
    return proof


def banks(proof):
    with np.load(DATA_ROOT/'data/proteus_windows.npz', allow_pickle=False) as z:
        meta = {k: z[k] for k in ('subject', 'session', 'interface')}
    meta['stream'] = np.char.add(np.char.add(meta['subject'], '|'), meta['session'])
    people = np.array([p.rsplit('/', 1)[-1] for p in meta['subject']])
    streams, rows_by_person = {}, {}
    with np.load(BANK/'banks.npz', allow_pickle=False) as z:
        for person in COHORT:
            rows = z[person+'_rows']
            items = z[person+'_support']
            expected = proof['data'][person]
            require(np.array_equal(rows, np.flatnonzero(people == person))
                    and digest(rows) == expected['row_sha'] and digest(items) == expected['support_sha']
                    and items.shape == (len(rows), 1152) and np.isfinite(items).all(), 'bank row/support differs')
            rows_by_person[person] = rows
            data = {}
            for order in ORDERS:
                fixed = z[f'{person}_{order}_current']
                delivery = z[f'{person}_{order}_delivery']
                require(fixed.shape == (len(rows), 3) and np.isfinite(fixed).all() and (fixed > 0).all()
                        and len(delivery) == len(rows) and np.array_equal(np.sort(delivery), np.arange(len(rows))),
                        'bank order or A6 probability differs')
                data[order] = (fixed[delivery], items[delivery], delivery)
            streams[person] = data
    return streams, rows_by_person, meta, people


def fit_many(episodes, prefix):
    batches, targets = [], []
    for streams, y in episodes:
        probability, items = stack(streams)
        batches.append((probability.log(), items))
        targets.extend(y[streams[order][2]] for order in ORDERS)
    targets = torch.from_numpy(np.concatenate(targets).astype(np.int64))
    counts = torch.bincount(targets, minlength=3)
    require((counts > 0).all(), 'readout source class missing')
    weights = targets.numel() / (3 * counts.double())
    torch.manual_seed(357)
    model = Readout().double()
    initial = torch.cat([p.detach().ravel() for p in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    def loss():
        predicted = torch.cat([model(logp, item, prefix).reshape(-1, 3) for logp, item in batches])
        return F.cross_entropy(predicted, targets, weight=weights)

    max_gradient = 0.
    for _ in range(16):
        cost = loss()
        require(torch.isfinite(cost), 'nonfinite readout source loss')
        optimizer.zero_grad()
        cost.backward()
        max_gradient = max(max_gradient, float(model.item[0].weight.grad.abs().max()))
        optimizer.step()
    with torch.no_grad():
        shift = float((torch.cat([p.ravel() for p in model.parameters()]) - initial).abs().max())
        final = float(loss())
    require(max_gradient > 1e-8 and shift > 1e-7 and np.isfinite(final), 'readout fitting silent no-op')
    return model.eval(), dict(steps=16, initial_sha=digest(initial.numpy()), parameters=len(initial),
                              max_gradient=max_gradient, max_parameter_change=shift, source_only_loss=final)


def smoke(streams, source_rows, total_rows):
    y = labels(source_rows, total_rows)
    for prefix in (False, True):
        actual, _ = fit_many([(streams, y)], prefix)
        original, _ = original_fit(streams, y, prefix)
        require(all(torch.equal(got, reference) for got, reference in zip(actual.parameters(), original.parameters())),
                'generalized source recipe differs from the original fit')


def fold(held, streams, rows, total_rows):
    fit_people = tuple(person for person in COHORT if person != held)
    require(len(fit_people) == 3 and all(not np.intersect1d(rows[held], rows[p]).size for p in fit_people),
            'held person in readout fitting')
    episodes = [(streams[p], labels(rows[p], total_rows)) for p in fit_people]
    current, control = fit_many(episodes, prefix=False)
    prefix, action = fit_many(episodes, prefix=True)
    require(control['parameters'] == action['parameters'] == 18819
            and control['initial_sha'] == action['initial_sha'], 'matched readout capacity mismatch')
    matched_rows = min(len(rows[held]), len(rows[fit_people[0]]))
    # The original interleave helper pairs an equal count of actual rows.
    guard_streams = {o: tuple(v[:matched_rows] for v in stream) for o, stream in streams[held].items()}
    checks = guards(current, prefix, guard_streams, streams[fit_people[0]])
    checks['interleaved_rows'] = matched_rows
    outputs, changes = {}, 0
    for order in ORDERS:
        p, item, delivery = streams[held][order]
        probability, x = torch.from_numpy(p).double(), torch.from_numpy(item).double()
        reset = np.argsort(delivery)
        outputs[held, order, 'A6'] = p[reset]
        for name, model, use_prefix in (('current', current, False), ('prefix', prefix, True)):
            emitted = model.replay(probability, x, prefix=use_prefix)
            with torch.no_grad():
                vector = model(probability.log()[None], x[None], use_prefix).softmax(-1)[0]
            require(torch.allclose(emitted, vector, atol=1e-12, rtol=0),
                    'complete held stream differs from the original vectorized fit path')
            outputs[held, order, name] = emitted.numpy()[reset]
        changes += int(np.sum(outputs[held, order, 'prefix'].argmax(1) != outputs[held, order, 'current'].argmax(1)))
    require(changes > 0, 'prefix arm had no decision-level changes in this fold')
    return outputs, dict(fit_people=list(fit_people), held_rows=digest(rows[held]),
        fit_rows={person: digest(rows[person]) for person in fit_people}, control=control, prefix=action,
        active_decision_changes=changes, guards=checks, held_labels_opened_in_fold=False)


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    proof = verify()
    streams, rows, metadata, people = banks(proof)
    smoke(streams['P02'], rows['P02'], len(people))
    print('original one-person source recipe BITWISE smoke', flush=True)
    outputs, traces = {}, {}
    for held in COHORT:
        values, traces[held] = fold(held, streams, rows, len(people))
        outputs.update(values)
        print(f'{held} all eight label-blind order folds pass', flush=True)
    require(len(outputs) == 4*8*3, 'incomplete three-arm, eight-order four-fold matrix')
    verify()
    np.savez_compressed('outputs.npz', **{'|'.join(key): value for key, value in outputs.items()})
    before = dict(bank_sha256=proof['banks_sha256'], body_sha256=proof['body_sha256'],
                  source_sha256=sha(Path('source.sha256')), output_sha256=sha(Path('outputs.npz')),
                  source_smoke_bitwise=True, completed_cells=len(outputs), each_fold_query_labels_excluded=True,
                  scoring_labels_opened=False, traces=traces)
    with Path('before_labels.json').open('x') as f:
        json.dump(before, f, indent=2)
    official_scorer = official()
    pin(Path(official_scorer.__file__), METRIC_SHA)
    result = score(official_scorer, labels, metadata, people, COHORT, ORDERS, outputs)
    result.update(scope='fixed A6 whole-readout, four exposed body-excluded people; no system/returning/sealed',
                  before_labels_sha256=sha(Path('before_labels.json')), metric_sha256=METRIC_SHA,
                  orders=list(ORDERS), script_wall_s=time.monotonic()-start,
                  rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with Path('result.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result['contrasts']), flush=True)


if __name__ == '__main__':
    main()
