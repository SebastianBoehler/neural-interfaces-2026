"""Score the sealed eight-order returning bank with official cell weights."""
import json
from pathlib import Path
import numpy as np
from fusion_inputs import official
import score379 as ref
from produce import DATA, PEOPLE, PRIOR, SEEDS, load_json

groups = (np.arange(4),np.arange(4,6),np.arange(6,8))


def summarize(values):
    people=sum(values[:,g].mean(1) for g in groups)/3
    rng=np.random.default_rng(457)
    pi=rng.integers(2,size=(4000,2))
    weights=np.array([.2,.8])[pi]
    draws=np.zeros(pi.shape)
    for g in groups:
        oi=rng.choice(g,size=(*pi.shape,len(g)))
        draws+=values[pi[:,:,None],oi].mean(2)/3
    sampled=(weights*draws).sum(1)/weights.sum(1)
    return dict(people=dict(zip(PEOPLE,people.tolist())),mix=float(people@[.2,.8]),
                paired80=np.percentile(sampled,[10,90]).tolist())


def main():
    bank={}
    for order in ref.ORDERS:
        path=Path(order+'.npz')
        lock=json.loads(path.with_suffix('.json').read_text())
        assert lock['source']==ref.sha(Path('source.sha256')) and lock['blind']==ref.sha(path)
        with np.load(path) as f:
            bank[order]={k:f[k] for k in f.files}
    assert sum(len(a['rows']) for a in bank.values())==6096 and ref.sha(DATA)==ref.DATA_SHA
    with np.load(DATA) as f:
        meta={k:f[k] for k in ('subject','session','interface')}
        y=f['y']
    meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'].astype(str))
    tables={k:np.empty((2,8)) for k in bank['within0'] if k not in ('rows','recognized')}
    by_route={k:{'before':0,'recognized':0} for k in tables}
    metric=official()
    for oi,order in enumerate(ref.ORDERS):
        a=bank[order]
        rows=a['rows']
        for k,table in tables.items():
            changed=(a[k]!=a['stock'])
            for key,mask in [('before',~a['recognized']),('recognized',a['recognized'])]:
                by_route[k][key]+=int(sum(changed&mask))
            for pi,p in enumerate(PEOPLE):
                include=(meta['subject']=='Dreyer2026Proteus/'+p)&(meta['session'].astype(str)!='1')
                idx=np.flatnonzero(include)
                pos=np.searchsorted(rows,idx)
                assert np.array_equal(rows[pos],idx)
                score=metric(y[idx],a[k][pos],meta,include)
                assert score['n_cells']==4
                table[pi,oi]=100*score['balanced_accuracy_cell_mean']
    exact=load_json(PRIOR/'result.json','77b4579e92a5f26036da14253c4e328c9df5b71f2930bd8e656bec5250a49b9a')
    assert np.allclose(tables['stock'],exact['person_order'],rtol=0,atol=1e-8)
    contrasts={}
    for seed in SEEDS:
        for arm,b in [('memory','stock'),('protected','stock'),('memory',f'{seed}_donor')]:
            k=f'{seed}_{arm}'
            contrasts[k+'-'+b]=summarize(tables[k]-tables[b])
    result={'scope':'conditional returning historical full-system dev, P04/P10, eight seen-session deliveries',
        'source':ref.sha(Path('source.sha256')),'absolute':{k:summarize(t) for k,t in tables.items()},
        'changes':by_route,'contrasts':contrasts,'person_order':{k:v.tolist() for k,v in tables.items()},
        'inputs':{o:ref.sha(Path(o+'.npz')) for o in ref.ORDERS}}
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(contrasts)


if __name__=='__main__':
    main()
