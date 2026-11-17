"""Prepare four honest A6 proxy pairs and their original complete-system target pairs."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
OLD = ROOT/'ni26_refit375_v1_20261117'
BANK = ROOT/'ni26_prefix358_v2_20261117'
sys.path.insert(0, str(OLD))
from trajectory import INPUT, SCHEDULES, check, digest, prepare, replay

PEOPLE = ('P01', 'P02', 'P03', 'P09')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def verify():
    pins = {BANK/'proof.json': 'a16b5715ef7a53c0052f1d789f638d62a7cfff2924d04a21f252a3dac9fc0ad7',
            OLD/'source.sha256': '289a7f6d7540ddaa4db1f42db5a1f67a323fcc2626057e94a5bdd76694404b1c',
            OLD/'before_labels.json': '5d59dbc223ed6a4d9e454d91c44b0fc223955a6a716bac874020a5fe4c1d4afa',
            OLD/'features.npy': 'fcbd25192a1e1a030c0905f2cc795c21e1984cbb7ff9c1fbd4b5653e21ec9103'}
    for path, wanted in pins.items():
        check(digest(path) == wanted, f'snapshot {path}')
    for line in (OLD/'source.sha256').read_text().splitlines():
        h, name = line.split(maxsplit=1)
        check(digest(OLD/name) == h, '375 source changed')
    proof = json.loads((BANK/'proof.json').read_text())
    check(not proof['label_accessed'] and tuple(proof['cohort']) == PEOPLE
          and set(proof['body_train_people']).isdisjoint(PEOPLE)
          and proof['banks_sha256'] == digest(BANK/'banks.npz'), 'A6 bank proof')


def features(prefix):
    check(prefix.shape == (100, 3) and (prefix > 0).all() and np.isfinite(prefix).all(), 'invalid prefix')
    p = prefix / prefix.sum(axis=1, keepdims=True)
    return np.array([p.max(axis=1).mean(), -(p * np.log(p)).sum(axis=1).mean(),
                     .5 * np.abs(p[60:80].mean(0) - p[80:100].mean(0)).sum()])


def produce(person, smoke):
    verify()
    assets, cls, cfg = prepare()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        subjects = np.array([s.split('/')[-1] for s in z['subject']])
    feat = np.load(OLD/'features.npy', mmap_mode='r')
    previous = json.loads((OLD/'before_labels.json').read_text())
    result = {}
    with (np.load(BANK/'banks.npz') as src, np.load(INPUT/'inputs.npz') as ctl,
          np.load(INPUT/'outputs.npz') as native):
        allrows = src[person+'_rows']
        check(np.array_equal(allrows, np.flatnonzero(subjects == person)), 'body-excluded rows')
        for oi, order in enumerate(ORDERS):
            key = order+'|'
            rows = allrows[src[f'{person}_{order}_delivery']]
            indices, bounds = ctl[key+'index'], ctl[key+'bounds']
            matches = [sid for sid, r in enumerate(np.split(indices, bounds[1:-1])) if np.array_equal(r, rows)]
            check(len(matches) == 1, 'A6 and target order differ')
            p = src[f'{person}_{order}_current']
            proxy = np.full((len(subjects), 3), np.nan)
            proxy[allrows] = p
            f, d = [replay(cls, cfg, rows, feat, proxy, proxy, sched) for sched in SCHEDULES]
            check(np.array_equal(f[:100], d[:100]), 'premature action')
            if smoke:
                r = replay(cls, cfg, rows[:131], feat, proxy, proxy, SCHEDULES[1])
                check(np.array_equal(r, d[:131]), 'reset/prefix')
            base = sorted(set(subjects)).index(person)
            path = OLD/f'shard_{2*base+oi%2}.npz'
            check(digest(path) == previous['shards'][path.name], 'target trajectory identity')
            with np.load(path) as saved:
                k = f'{order}|{matches[0]}|'
                fixed, delayed = [saved[k+a] for a in ('fixed', 'delayed')]
                check(np.array_equal(saved[k+'rows'], rows)
                      and np.array_equal(fixed, ctl[key+'expected'][rows].argmax(1))
                      and np.array_equal(fixed, native[key+'stock'][rows].argmax(1))
                      and np.array_equal(fixed[:100], delayed[:100]), 'target frozen/native/prefix')
            result.update({key+'rows': rows, key+'proxy': np.stack([f.argmax(1), d.argmax(1)]),
                           key+'target': np.stack([fixed, delayed]),
                           key+'proxy_features': features(proxy[rows[:100]]),
                           key+'target_features': features(ctl[key+'raw'][rows[:100]])})
            if smoke:
                Path('smoke.json').write_text(json.dumps({'source': digest('source.sha256'), 'person': person}))
                return
    np.savez_compressed(person+'.npz', **result)


if __name__ == '__main__':
    produce(PEOPLE[int(sys.argv[2])], sys.argv[1] == 'smoke')
