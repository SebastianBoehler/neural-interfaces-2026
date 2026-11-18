"""Attribute independent stock drift and report exact shared-input evidence."""
import hashlib
import json
from pathlib import Path

ARMS = ('stock1','stock2')
FIELDS = ('stock','base','filtered','window','ft_count','ft_tail','ft_weight')


def load(arm):
    p = Path(arm+'.json')
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == p.with_suffix('.sha256').read_text().strip()
    return json.loads(p.read_bytes()),h


def main():
    (a,ha),(b,hb) = map(load,ARMS)
    assert all(a[k] == b[k] for k in ('rows','labels','activity','stock_cov','deferred_cov','source'))
    assert all(t['stock']==t['deferred'] for record in (a,b) for t in record['trace'])
    traces = list(zip(a['trace'],b['trace']))
    indices = {k:[i for i,(x,y) in enumerate(traces) if x[k]!=y[k]] for k in FIELDS}
    deltas = [max(abs(c-d) for c,d in zip(x['values'],y['values'])) for x,y in traces]
    score, filtered = indices['stock'], indices['filtered']
    attribution = bool(score and score[0]>=20 and max(deltas)<2e-5 and
                       (score[0]>=90 or score[0] in filtered))
    result = dict(shared_input_bitwise_per_pass=len(traces),labels_exact=True,
        stock_cov=a['stock_cov'],deferred_cov=a['deferred_cov'],matched_activity=a['activity'],
        independent_stock={k:dict(n=len(v),first=v[0] if v else None) for k,v in indices.items()},
        max_independent_stock_abs=max(deltas),preregistered_attribution=attribution,
        source=a['source'],inputs=dict(zip(ARMS,(ha,hb))))
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__ == '__main__':
    main()
