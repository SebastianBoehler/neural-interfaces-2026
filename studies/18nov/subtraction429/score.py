"""Score locked predictions only after checking source, order and prefix identities."""
import hashlib
import json
from pathlib import Path

import numpy as np

from wang_inputs import orders


PEOPLE = range(2, 8)
ARMS = ('A6', 'pop', 'old', 'cap')
TASKS = ('subtraction', 'memory', 'lyrics')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    records = np.full((6, 2, 3, 4, 3, 3), np.nan)
    ranks = []
    rms = []
    digest = hashlib.sha256()
    refits = []
    for i, person in enumerate(PEOPLE):
        tag = f'sub-{person:02d}'
        src = read(tag+'.json')
        assert len(src['ranges']) == 9
        rms.append(src['relative_rms'])
        with np.load(tag+'.npz') as z:
            ranks.append(z['ranks'])
        for group in ('a6', 'old', 'cap'):
            name = f'{group}_{person}.json'
            digest.update(Path(name).read_bytes())
            for r in read(name):
                v = r['visit']
                j = (429,1429).index(r['seed'])
                k = ARMS.index(r['arm'])
                expected = dict(orders(person, v))[r['seed']]
                assert np.array_equal(r['rows'], expected)
                labels = np.asarray(r['labels'])
                assert np.isin(labels, [0,1,2]).all() and labels.shape == (18,)
                cell = records[i,j,v-1,k]
                assert np.isnan(cell).all()
                for task in range(3):
                    cell[task] = 100*np.bincount(labels[expected//6 == task], minlength=3)/6
                if group == 'cap':
                    refits.append(r['activity']['ft_refits'])
                if person == 2 and v == 1 and not j:
                    smoke, = [s for s in read(group+'_smoke.json') if s['arm'] == r['arm']]
                    assert smoke['rows'] == r['rows'] and smoke['labels'] == r['labels']
    assert np.isfinite(records).all() and len(refits) == 36
    assert np.array_equal(np.array(ranks)[:,:,0], np.tile([5,4,3], (6,1)))
    assert np.max(np.abs(records.sum(-1)-100)) < 1e-8
    rng = np.random.default_rng(429)
    persons = rng.integers(0,6,(10000,6))
    order = rng.integers(0,2,(10000,6,2))

    def stat(x):
        assert x.shape == (6,2)
        boots = x[persons[:,:,None],order].mean((1,2))
        return dict(mean=float(x.mean()), ci80=np.quantile(boots,[.1,.9]).tolist(),
                    per_person=x.mean(1).tolist())

    out = dict(people=list(PEOPLE), arms={}, source_rms=rms, source_qualified=max(rms)<.01,
               cap_refits=sum(refits), predictions_sha256=digest.hexdigest())
    sub = records[...,1]
    avg = sub[...,0] - sub[...,1:].mean(-1)
    for k, arm in enumerate(ARMS):
        part = sub[:,:,:,k,:]
        earliest = np.broadcast_to(np.array(ranks).argmin(-1)[:,None,:,None], (6,2,3,1))
        latest = np.broadcast_to(np.array(ranks).argmax(-1)[:,None,:,None], (6,2,3,1))
        control = np.take_along_axis(part, latest, axis=-1) - np.take_along_axis(part,earliest,axis=-1)
        out['arms'][arm] = dict(absolute_rates=records[:,:,:,k].mean((0,1,2)).tolist(),
            against_mean=stat(avg[:,:,:,k].mean(2)),
            against_memory=stat((part[...,0]-part[...,1]).mean(2)),
            against_lyrics=stat((part[...,0]-part[...,2]).mean(2)),
            visit=[stat(avg[:,:,v,k]) for v in range(3)],
            last_minus_first=stat(control[...,0].mean(2)))
    for key, i, j in [('cap_minus_old',3,2), ('a6_minus_pop',0,1)]:
        out[key] = stat((avg[:,:,:,i]-avg[:,:,:,j]).mean(2))
    Path('result.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
