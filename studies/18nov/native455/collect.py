"""Score committed bank cells, with people and order uncertainty from §443."""
import json
from pathlib import Path
import sys
import numpy as np
import score379 as ref
from fusion_inputs import official
from score import interval

CONTRASTS = {'prior': [('prior','stock')], 'reliability': [('reliable','stock'),('reliable','cyclic'),('stack','prior')]}


def main(phase):
    pairs = CONTRASTS[phase]
    arms = sorted({a for pair in pairs for a in pair} | {'stock'})
    banks = {}
    for o in ref.ORDERS:
        path = Path(o+'.npz')
        proof = json.loads(path.with_suffix('.json').read_text())
        assert proof['bank'] == ref.sha(path) and proof['source'] == ref.sha(Path('source.sha256'))
        with np.load(path) as z:
            banks[o] = {k:z[k] for k in arms}
    path = ref.BASE/'data/proteus_windows.npz'
    assert ref.sha(path) == ref.DATA_SHA
    with np.load(path) as z:
        y = z['y']
        meta = {k:z[k] for k in ('subject','session','interface')}
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    people = np.array([s.rsplit('/',1)[-1] for s in meta['subject']])
    metric = official()
    tables = {a:np.empty((10,8)) for a in arms}
    counts = []
    for pi,p in enumerate(ref.PEOPLE):
        mask = people==p
        counts.append(len(set(zip(meta['stream'][mask],meta['interface'][mask]))))
        for i,o in enumerate(ref.ORDERS):
            for a in arms:
                m = metric(y[mask],banks[o][a][mask],meta,mask)
                assert m['n_cells'] == counts[-1]
                tables[a][pi,i] = 100*m['balanced_accuracy_cell_mean']
    counts = np.array(counts)
    absolute = {a:interval(t, counts)['pp'] for a,t in tables.items()}
    assert abs(absolute['stock']-76.2793)<.00005 and counts.sum()==28
    contrasts = {}
    for a,b in pairs:
        delta=tables[a]-tables[b]
        contrasts[a+'-'+b] = dict(**interval(delta,counts),
            orders=dict(zip(ref.ORDERS,np.average(delta,axis=0,weights=counts))),
            changed=sum(int((v[a]!=v[b]).sum()) for v in banks.values()))
    Path(phase+'_result.json').write_text(json.dumps(dict(absolute=absolute,contrasts=contrasts,
        source=ref.sha(Path('source.sha256')),banks={o:ref.sha(Path(o+'.npz')) for o in ref.ORDERS}),indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
