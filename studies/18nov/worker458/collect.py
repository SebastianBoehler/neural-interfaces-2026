import json
from pathlib import Path
from statistics import median


def collect(person):
    pairs=[]
    for rep in range(2):
        records={arm:json.loads(Path(f'{person}_{rep}_{arm}.json').read_text()) for arm in ('stock','graph')}
        stock,graph=records.values()
        for k in ('session','windows','counters','device'):
            assert stock[k]==graph[k]
        assert stock['labels']['stock']==graph['labels']['stock']
        pairs.append(dict(**records,ratio=graph['seconds']/stock['seconds']))
    assert pairs[0]['graph']['labels']==pairs[1]['graph']['labels']
    return dict(pairs=pairs,median_ratio=median(p['ratio'] for p in pairs))


if __name__=='__main__':
    Path('result.json').write_text(json.dumps({p:collect(p) for p in (0,5)},indent=2))
