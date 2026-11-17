"""Four body-excluded A6 banks, with no dataset label access."""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

READY = Path('/mnt/lustre/work/gehler/gfh527/ni26_prefix357_v1_20261117')
sys.path.insert(0, str(READY))
from inputs import (EXTEND, FIT_PEOPLE, ORDERS, bank, digest, extract, permutations,
                    pin, preflight, require, sha, state_sha)
from common346 import CONTROL, native
from recipe336 import deep, load as load_metadata

COHORT = ('P01', 'P02', 'P03', 'P09')
READY_SOURCE = 'ff2087149a43aab176233b78a7325939d84b694befd488690ed85d5ae7197b06'
READY_PROOF = '09d7a09823249084b84cc95b39e9aa524ecf7f111c1a63abbb7483216be51518'
BODY_SHA = 'eadd5c23c9f7cac239bf78a99993d8d3e47392d24d166e0ede6fce5b0e6ac9f8'
BODY_RECORD_SHA = '701ee6dbba3fb0310beebf9b2f38d9bb6bf5f07e5794b061358108cfd518ee2e'


def verify():
    pin(READY/'source.sha256', READY_SOURCE)
    pin(READY/'proof.json', READY_PROOF)
    for line in (READY/'source.sha256').read_text().splitlines():
        hexdigest, name = line.split(maxsplit=1)
        pin(READY/name, hexdigest)
    proof = json.loads((READY/'proof.json').read_text())
    archived = preflight()
    pin(CONTROL/'record.json', BODY_RECORD_SHA)
    body = json.loads((CONTROL/'record.json').read_text())
    require(proof['status'] == 'SOURCE_ONLY_READINESS_PASS'
            and body['checkpoint_sha256'] == proof['body_sha'] == archived['checkpoint_sha256'] == BODY_SHA,
            'A6 body/readiness identity changed')
    pin(CONTROL/'model.pt', BODY_SHA)
    pin(CONTROL/'predictions.npz', body['prediction_sha256'])
    return body


def restore(data, training_rows, body):
    require('y' not in data and body['source_rows_sha256'] == digest(training_rows), 'body source hash/mask changed')
    model = native(data)
    checkpoint = torch.load(CONTROL/'model.pt', map_location='cpu', weights_only=False)
    source_covariance = model.C[training_rows].mean(0)
    scale = float(np.sqrt(np.trace(source_covariance) / len(source_covariance)))
    require(model.align is False and not model.ref_clip and not model.ref_band and not model.crops
            and np.array_equal(source_covariance, checkpoint['G']) and scale == checkpoint['scale'],
            'label-free restore differs from source A6 preprocessing')
    model.G, model.scale = checkpoint['G'], checkpoint['scale']
    model.net = deep.build(model.arch, 41, 480, model.pretrained, **model.net_kw).to(model.dev)
    model._prepare()
    model.net.load_state_dict(checkpoint['state'], strict=True)
    model.net.eval()
    model.set_k(20.)
    require(len(model.sbns()) == 1 and model.dev == 'cpu', 'A6 inference path changed')
    return model


def build():
    started = time.monotonic()
    torch.set_num_threads(2)
    body = verify()
    data, _, _, people = load_metadata()
    require('y' not in data, 'unexpected label member')
    mask = np.flatnonzero(np.isin(people, FIT_PEOPLE))
    require(tuple(sorted(set(people[mask]))) == FIT_PEOPLE
            and tuple(sorted(set(people) - set(FIT_PEOPLE))) == COHORT, 'cohort is not fixed body-complement')
    model = restore(data, mask, body)
    state = state_sha(model.net)
    permutations_by_order = permutations(len(people))
    out, records, rows_by_person = {}, {}, {}
    with (
        np.load(EXTEND/'predictions.npz', allow_pickle=False) as pair,
        np.load(CONTROL/'predictions.npz', allow_pickle=False) as p09,
    ):
        for person in COHORT:
            start = time.monotonic()
            rows = np.flatnonzero(people == person)
            rows_by_person[person] = rows
            require(len(set(data['stream'][rows])) == 1 and not np.intersect1d(mask, rows).size,
                    'fixed held session partition changed')
            if person in ('P02', 'P03'):
                require(np.array_equal(pair[f'{person}_index'], rows), 'archived pair row alignment failed')
            if person == 'P09':
                require(np.array_equal(p09['index'], rows), 'archived P09 row alignment failed')
            model.set_k(None)
            items = np.concatenate([extract(model, np.asarray([row]))[0] for row in rows])
            last_first = np.concatenate([extract(model, np.asarray([row]))[0] for row in rows[:3][::-1]])
            require(np.array_equal(last_first[::-1], items[:3]), 'pointwise support order guard failed')
            model.set_k(20.)
            out.update({person+'_rows': rows, person+'_support': items})
            records[person] = dict(rows=len(rows), row_sha=digest(rows), support_sha=digest(items), orders={})
            for order in ORDERS:
                perm = permutations_by_order[order]
                _, probabilities, guard = bank(model, data, rows, perm)
                delivered = perm[np.isin(perm, rows)]
                positions = np.searchsorted(rows, delivered)
                require(np.array_equal(np.sort(positions), np.arange(len(rows)))
                        and np.array_equal(delivered, rows[positions])
                        and probabilities.shape == (len(rows), 3) and np.isfinite(probabilities).all()
                        and (probabilities > 0).all() and np.allclose(probabilities.sum(1), 1., atol=1e-12, rtol=0),
                        'actual stream order/probability layout changed')
                if person != 'P01':
                    archived = p09[order] if person == 'P09' else pair[f'{person}_A_{order}']
                    error = float(np.max(np.abs(probabilities - archived)))
                    require(error <= 1e-6 and np.array_equal(probabilities.argmax(1), archived.argmax(1)),
                            f'{person}/{order} causal current archive mismatch {error:.8g}')
                    guard['archived_probability_max_abs'] = error
                out.update({f'{person}_{order}_current': probabilities, f'{person}_{order}_delivery': positions})
                records[person]['orders'][order] = guard
            records[person]['wall_s'] = time.monotonic() - start
            print(f'{person}: eight current orders and B1 support guards passed', flush=True)
    for person in COHORT:
        _, repeat, _ = bank(model, data, rows_by_person[person], permutations_by_order['within0'])
        require(np.array_equal(repeat, out[f'{person}_within0_current']), 'cross-person reset failed')
    require(state == state_sha(model.net), 'A6 mutated in the bank build')
    verify()
    np.savez_compressed('banks.npz', **out)
    proof = dict(status='FOUR_BODY_EXCLUDED_PEOPLE_LABEL_FREE_BANKS_PASS', cohort=list(COHORT),
                 body_train_people=list(FIT_PEOPLE), body_sha256=BODY_SHA,
                 label_accessed=False, optimizer_steps=0, quality_scored=False, orders=list(ORDERS),
                 data=records, banks_sha256=sha(Path('banks.npz')), source_sha256=sha(Path('source.sha256')),
                 elapsed_s=time.monotonic()-started, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with Path('proof.json').open('x') as stream:
        json.dump(proof, stream, indent=2)
    print(proof['status'], flush=True)


if __name__ == '__main__':
    build()
