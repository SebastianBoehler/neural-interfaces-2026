"""Score a fixed terminal fusion only after committing the decisions for all eligible rows."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from fusion_inputs import (BRANCHES, CHAIN, COHORT, FITS, MEMBERS, WEIGHTS, WORK,
                          member_bank, official, pin, probability, restore, sha)


def build(seed,smoke):
    sc,scores,meta,ages=restore()
    orders=sc.ORDERS[:1] if smoke else sc.ORDERS
    assert sc.BASES[0]=='B2_pool' and len(sc.BASES)==6
    result={}
    changed={}
    for person in COHORT:
        rows=np.flatnonzero(meta['person']==person)
        bank=member_bank(seed,person,rows,orders)
        for branch in BRANCHES:
            ref='frozen' if branch=='control' else 'JS'
            for order in orders:
                for base in sc.BASES:
                    source=scores[branch,order,base][rows]
                    fixed=probability(source,ages[order][rows])
                    original=source.argmax(1).astype(np.uint8)
                    if base=='B2_pool':
                        assert np.array_equal(original,bank[ref,order].argmax(1))
                    key=(person,branch,order,base)
                    result[key+('zero',)]=original
                    for arm in MEMBERS:
                        p=bank[arm,order]
                        assert np.array_equal((fixed+0*p).argmax(1),original)
                        for w in WEIGHTS:
                            name=f'{arm}_{int(100*w)}'
                            pred=((1-w)*fixed+w*p).argmax(1).astype(np.uint8)
                            result[key+(name,)]=pred
                            count=f'{branch}|{base}|{name}'
                            changed[count]=changed.get(count,0)+int((pred!=original).sum())
    assert any(v>0 for v in changed.values()),'no decision intervention'
    return sc,meta,result,changed


def interval(delta):
    groups=(np.arange(4),np.arange(4,6),np.arange(6,8))
    rng=np.random.default_rng(381)
    draws=rng.integers(4,size=(4000,4))
    values=np.mean([delta[:,g].mean(1) for g in groups],axis=0)
    boots=np.zeros((4000,4))
    for g in groups:
        oi=rng.choice(g,size=(4000,4,len(g)))
        boots+=delta[draws[:,:,None],oi].mean(2)/3
    return dict(pp=float(values.mean()),paired80=np.percentile(boots.mean(1),[10,90]).tolist(),
                people=dict(zip(COHORT,values.tolist())))


def score(sc,meta,out):
    metric=official()
    with np.load(CHAIN/'data/proteus_windows.npz') as d:
        y=d['y']
    names=['zero']+[f'{a}_{int(100*w)}' for a in MEMBERS for w in WEIGHTS]
    vals={b:{a:np.empty((4,8,6)) for a in names} for b in BRANCHES}
    for pi,p in enumerate(COHORT):
        mask=meta['person']==p
        for bi,b in enumerate(sc.BASES):
            for oi,o in enumerate(sc.ORDERS):
                for branch in BRANCHES:
                    for arm in names:
                        pred=out[p,branch,o,b,arm]
                        r=metric(y[mask],pred,meta,mask)
                        assert r['n_cells']==2
                        vals[branch][arm][pi,oi,bi]=100*r['balanced_accuracy_cell_mean']
    report={}
    for branch in BRANCHES:
        for base in ('B2_pool','six_base_mean'):
            values={k:(v[:,:,0] if base=='B2_pool' else v.mean(2)) for k,v in vals[branch].items()}
            zero=interval(values['zero'])['pp']
            if base=='B2_pool':
                expect=81.21561492579204 if branch=='control' else 81.23552194696784
                assert abs(zero-expect)<1e-9
            contrasts={}
            for w in WEIGHTS:
                suffix='_'+str(int(100*w))
                for a in MEMBERS:
                    name=a+suffix
                    contrasts[name+'-zero']=interval(values[name]-values['zero'])
                    if a!='J_real':
                        j='J_real'+suffix
                        contrasts[j+'-'+name]=interval(values[j]-values[name])
            report[branch+'|'+base]=dict(zero=zero,contrasts=contrasts)
    return report


def main(arg):
    lock=sha(Path('source.sha256'))
    if arg=='smoke':
        guards={}
        for seed in FITS:
            _,_,pred,changed=build(seed,True)
            guards[seed]=dict(pinned_decisions=len(pred),changed_counts=sum(changed.values()))
        Path('smoke.json').write_text(json.dumps(dict(source=lock,checked=guards,labels_opened=False)))
        return
    seed=int(arg)
    guard=json.loads(Path('smoke.json').read_text())
    assert guard['source']==lock and guard['labels_opened'] is False
    sc,meta,pred,changed=build(seed,False)
    out=Path(str(seed)+'.npz')
    np.savez_compressed(out,**{'|'.join(k):v for k,v in pred.items()})
    before=dict(source=lock,seed=seed,initial_fit_proof=FITS[seed][1],predictions=sha(out),
                changed=changed,labels_opened=False)
    before_path=Path(f'{seed}_before_labels.json')
    before_path.write_text(json.dumps(before,indent=2))
    data=score(sc,meta,pred)
    Path(f'{seed}_result.json').write_text(json.dumps(dict(seed=seed,before=sha(before_path),data=data),indent=2))
    print(seed,data['JS|B2_pool']['contrasts']['J_real_5-zero'],flush=True)


if __name__=='__main__':
    main(sys.argv[1])
