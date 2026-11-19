"""Summarize exact matched CPU score/work proof and session-paired timings."""
import json
from pathlib import Path

import numpy as np
import score379 as ref

GROUPS = (np.arange(4), np.arange(4,6), np.arange(6,8))


def summarize(records):
    people = sorted({stream.split('|')[0] for order in records for stream in order})
    by_role = {}
    for arm in ('stock', 'lazy'):
        time = np.asarray([[sum(np.median(v[f'{arm}_seconds']) for s,v in o.items()
                                if s.split('|')[0] == p) for o in records] for p in people])
        by_role[arm] = time
    ratios = by_role['lazy']/by_role['stock']
    if not np.isfinite(ratios).all() or not np.all(ratios>0):
        raise ValueError('invalid paired durations or missing person/orders')
    # Preserve equal delivery-scheme weight within each person.
    def typical(x):
        return np.median(sum(x[:,g].mean(1) for g in GROUPS)/3)
    rand = np.random.default_rng(465)
    counts = len(people)
    people_samples = rand.integers(counts,size=(4000,counts))
    boot = np.zeros(people_samples.shape)
    for g in GROUPS:
        orders = rand.choice(g,size=(*people_samples.shape,len(g)))
        boot += ratios[people_samples[:,:,None],orders].mean(2)/3
    stock, lazy = (sum(v[f'{arm}_cov'] for o in records for v in o.values())
                   for arm in ('stock','lazy'))
    return dict(score_delta_pp=0., covariance_stock=stock, covariance_lazy=lazy,
                savings=1-lazy/stock, median_ratio=typical(ratios), people=people,
                people_order_ratios=ratios.tolist(),
                paired80=np.quantile(np.median(boot,axis=1),[.1,.9]).tolist(),
                summed_seconds={a:float(v.sum()) for a,v in by_role.items()})


def collect():
    source = ref.sha(Path('source.sha256'))
    result = {}
    for role,total,streams in (('new',2655,14),('returning',762,4)):
        records, receipts = [], {}
        for order in ref.ORDERS:
            path = Path(f'{role}_{order}.json')
            item = json.loads(path.read_text())
            assert item['role'] == role and item['order'] == order and item['source'] == source
            assert item['exact'] and len(item['records']) == streams
            assert sum(x['rows'] for x in item['records'].values()) == total
            records.append(item['records'])
            receipts[order] = ref.sha(path)
        result[role] = dict(exact_scores=total*len(ref.ORDERS), receipts=receipts,**summarize(records))
    result['advance'] = (result['new']['savings'] >= .20 and result['returning']['savings'] >= .35
                         and min(x['median_ratio'] for x in result.values()) <= .90)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k: (v['savings'],v['median_ratio']) for k,v in result.items() if k != 'advance'},
          'advance',result['advance'],flush=True)


if __name__ == '__main__':
    collect()
