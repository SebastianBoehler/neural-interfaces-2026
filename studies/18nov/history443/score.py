"""Commit the terminal decisions for each seed, then score complete held streams."""
import json
from pathlib import Path
import sys

import numpy as np
import score379 as ref
from fusion_inputs import BRANCHES, official, probability, restore, sha

ARMS = ('stock','memory','blank','donor','absolute')


def build(seed, people):
    sc,scores,meta,ages = restore()
    controls,check_ages = ref.controls_and_age(meta)
    assert tuple(sc.ORDERS)==ref.ORDERS
    out = {}
    for person in people:
        rows = np.flatnonzero(meta['person']==person)
        path = Path(f'{seed}_{person}.npz')
        record = json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==record['output'] and sha(Path('source.sha256'))==record['source']
        with np.load(path) as z:
            assert np.array_equal(z['index'],rows)
            bank = {k:z[k].astype(np.float64) for k in ('memory','blank','donor','source')}
        for branch in BRANCHES:
            for i,order in enumerate(sc.ORDERS):
                assert np.array_equal(ages[order], check_ages[i])
                raw = scores[branch,order,'B2_pool'][rows]
                prob = probability(raw,ages[order][rows])
                stock = prob.argmax(1)
                assert np.array_equal(stock,controls['frozen' if branch=='control' else 'JS'][i,rows])
                with np.errstate(divide='ignore'):
                    baseline = np.log(prob)
                assert np.array_equal(stock,baseline.argmax(1))
                decisions = {'stock':stock}
                for arm in ('memory','donor','blank','source'):
                    delta = bank[arm][i]-bank['source'][i]
                    correction = .1*np.clip(delta-delta.mean(1,keepdims=True),-1,1)
                    pred = (baseline+correction).argmax(1)
                    if arm=='source':
                        assert np.all(correction==0) and np.array_equal(pred,stock)
                    else:
                        decisions[arm] = pred
                logits = bank['memory'][i]
                q = np.exp(logits-logits.max(1,keepdims=True))
                decisions['absolute'] = (.95*prob+.05*q/q.sum(1,keepdims=True)).argmax(1)
                for arm,pred in decisions.items():
                    out['|'.join((person,branch,order,arm))] = pred.astype(np.uint8)
    return meta,out


def interval(delta,counts):
    rng = np.random.default_rng(443)
    groups = (np.arange(4),np.arange(4,6),np.arange(6,8))
    values = sum(delta[:,g].mean(1) for g in groups)/3
    pp = rng.integers(len(counts),size=(4000,len(counts)))
    draws = np.zeros(pp.shape)
    for g in groups:
        oi = rng.choice(g,size=(*pp.shape,len(g)))
        draws += delta[pp[:,:,None],oi].mean(2)/3
    weights = counts[pp]
    bootstrap = (weights*draws).sum(1)/weights.sum(1)
    return dict(pp=float(np.average(values,weights=counts)),paired80=np.percentile(bootstrap,[10,90]).tolist(),
                people=dict(zip(ref.PEOPLE,values.tolist())))


def score(seed):
    meta,out = build(seed,ref.PEOPLE)
    dest = Path(f'{seed}_decisions.npz')
    np.savez_compressed(dest,**out)
    before = {'predictions':sha(dest),'source':sha(Path('source.sha256')),
              'actors':{p:sha(Path(f'{seed}_{p}.json')) for p in ref.PEOPLE}}
    Path(f'{seed}_before_labels.json').write_text(json.dumps(before,indent=2))
    with np.load(ref.BASE/'data/proteus_windows.npz') as z:
        labels=z['y']
    counts=[]
    metric=official()
    tables={b:{a:np.empty((10,8)) for a in ARMS} for b in BRANCHES}
    for pi,p in enumerate(ref.PEOPLE):
        mask=meta['person']==p
        counts.append(len(set(zip(meta['stream'][mask],meta['interface'][mask]))))
        for branch in BRANCHES:
            for oi,order in enumerate(ref.ORDERS):
                for arm in ARMS:
                    pred=out['|'.join((p,branch,order,arm))]
                    r=metric(labels[mask],pred,meta,mask)
                    assert r['n_cells']==counts[-1]
                    tables[branch][arm][pi,oi]=100*r['balanced_accuracy_cell_mean']
    counts=np.array(counts)
    result={}
    for b,t in tables.items():
        contrasts=[(a,'stock') for a in ARMS[1:]]+[('memory',a) for a in ('donor','blank','absolute')]
        scores={a:interval(v,counts)['pp'] for a,v in t.items()}
        changed={a:sum(int((out['|'.join((p,b,o,a))]!=out['|'.join((p,b,o,'stock'))]).sum())
                       for p in ref.PEOPLE for o in ref.ORDERS) for a in ARMS[1:]}
        assert abs(scores['stock']-(76.2793 if b=='control' else 76.5517))<.0001
        result[b]=dict(scores=scores,changed=changed,contrasts={f'{a}-{c}':interval(t[a]-t[c],counts)
                       for a,c in contrasts})
    Path(f'{seed}_result.json').write_text(json.dumps(dict(seed=seed,**before,data=result),indent=2))


if __name__=='__main__':
    if sys.argv[1]=='smoke':
        build(379,('P01',))
    else:
        score(int(sys.argv[1]))
