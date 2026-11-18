"""Summarize authenticated AB/BA timings without opening task labels."""
import json
from pathlib import Path

import numpy as np

from verify_pair import ROLES, digest, pair

SEEDS = (411,1411)
ARMS = ('stock','deferred')
CATEGORIES = ('mixed',*ROLES)


def collect():
    checks = {}
    timings = np.empty((3,2,2,2,3))
    raw = {}
    stock_cov, lazy_cov = np.zeros((2,2),int)
    n = 0
    for idx in range(6):
        i,j = divmod(idx,2)
        previous = None
        for direction, order in enumerate(('ab','ba')):
            files = [Path(f'run-{idx}-{order}-{arm}.json') for arm in ARMS]
            a,b = matched = pair(*files)
            assert a['index'] == idx and a['mode'] == 'produce'
            if previous is not None:
                assert all(a['records'][r][k] == previous['records'][r][k]
                           for r in ROLES for k in ('rows','labels','terminal','work','cov'))
            previous = a
            for arm, value in enumerate(matched):
                periods = [value['records'][role]['wall_s'] for role in ROLES]
                timings[i,j,direction,arm] = (sum(periods),*periods)
                raw[files[arm].name] = dict(load_s=value['load_s'],proc_s=value['process_s'],
                     rss_kib=value['rss_kib'],gpu_live_bytes=value['gpu_live_bytes'],
                     gpu_reserved_bytes=value['gpu_reserved_bytes'],gpu=value['gpu'],
                     streams={r:{k:v for k,v in value['records'][r].items()
                                 if k in ('wall_s','restore_s','first_s','p95_s','max_s','cov')}
                              for r in ROLES})
            if direction == 0:
                stock_cov += [a['records'][r]['cov'] for r in ROLES]
                lazy_cov += [b['records'][r]['cov'] for r in ROLES]
                n += sum(len(a['records'][r]['labels']) for r in ROLES)
            for file in files:
                checks[file.name] = digest(file)
    Path('timings_receipt.json').write_text(json.dumps(checks,indent=2)+'\n')
    ratios = timings[:,:,:,1]/timings[:,:,:,0]
    rng = np.random.default_rng(4685)
    people = rng.integers(3,size=(10000,3,1))
    orders = rng.integers(2,size=(10000,3,2))
    boot = np.median(ratios[people,orders],axis=(1,2,3))
    result = {}
    for i,name in enumerate(CATEGORIES):
        ratio = ratios[:,:,:,i]
        result[name] = dict(point=float(np.median(ratio)),
                            paired80=np.quantile(boot[:,i],[.1,.9]).tolist(),
                            ab=float(np.median(ratio[:,:,0])),
                            ba=float(np.median(ratio[:,:,1])),
                            stock_median_s=float(np.median(timings[:,:,:,0,i])),
                            deferred_median_s=float(np.median(timings[:,:,:,1,i])))
    p = result['mixed']
    qualify = p['point'] <= .99 and p['paired80'][1] < 1 and max(p['ab'],p['ba']) <= 1
    out = dict(summary=result,qualify=qualify,refutes_mixed=p['paired80'][0]>=1,
               n_unique_rows=n,covariance=dict(stock=stock_cov.tolist(),deferred=lazy_cov.tolist()),
               covariance_order=ROLES,orders=SEEDS,source=digest(Path('source.sha256')),
               before_timings=digest(Path('timings_receipt.json')),runs=raw)
    Path('result.json').write_text(json.dumps(out,indent=2)+'\n')
    print(result,qualify,flush=True)


if __name__ == '__main__':
    collect()
