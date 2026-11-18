"""Score ten complete leave-person-out whole decoders with matched official controls."""
import json
from pathlib import Path
import sys
import numpy as np
from common import ARMS,DATA,OLD,ORDERS,PEOPLE,SEEDS,WORK,digest,metadata,sha

sys.path.insert(0,str(OLD))
assert sha(OLD/'score379.py')=='5adfcb30ea9c88ae4f3bc2ef128b502dba5bfc16a05054bc4f57f894289dcbaa'
from score379 import controls_and_age,source_manifest,METRICS


def summary(delta,counts):
    rng=np.random.default_rng(389)
    pp=rng.integers(10,size=(4000,10))
    v=np.zeros((4000,10))
    means=[]
    for g in (np.arange(4),np.arange(4,6),np.arange(6,8)):
        ix=rng.choice(g,size=(4000,len(g)))
        v+=delta[ix[:,:,None],pp[:,None,:]].mean(1)/3
        means.append(delta[g].mean(0))
    by=np.mean(means,axis=0)
    ci=np.percentile(np.average(v,axis=1,weights=counts[pp]),[10,90])
    return dict(pp=float(np.average(by,weights=counts)),paired80=ci.tolist(),people=dict(zip(PEOPLE,by.tolist())))


def main():
    meta=metadata()
    base,_=controls_and_age(meta)
    n=len(meta['subject'])
    out={key:np.empty((8,n),np.int8) for key in
         (f'{f}|{s}|{a}' for f in ('whole','half') for s in SEEDS for a in ARMS)}
    proofs={}
    for p in PEOPLE:
        rows=np.flatnonzero(meta['person']==p)
        fit=np.flatnonzero(meta['person']!=p)
        for f in ('whole','half'):
            tag=f+'_'+p
            rec=json.loads(Path(tag+'.json').read_text())
            assert rec['held']==p and rec['test']==digest(rows) and rec['fit']==digest(fit)
            assert rec['source']==sha('source.sha256') and rec['inputs']==sha('inputs.json')
            assert rec['outputs']==sha(tag+'.npz') and rec['held_labels_opened'] is False
            assert rec['seeds']['389']!=rec['seeds']['1389']
            proofs[tag]=rec
            with np.load(tag+'.npz') as z:
                assert np.array_equal(z['index'],rows)
                for seed in SEEDS:
                    for a in ARMS:
                        for i,o in enumerate(ORDERS):
                            val=z[f'{seed}|{a}|{o}']
                            assert val.shape==(len(rows),) and np.isin(val,[0,1,2]).all()
                            out[f'{f}|{seed}|{a}'][i,rows]=val
        assert proofs['half_'+p]['seeds']==proofs['whole_'+p]['seeds']
    before=Path('before_labels.json')
    before.write_text(json.dumps(dict(actors=proofs),indent=2))
    assert source_manifest(METRICS/'source.sha256')=='9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd'
    sys.path.insert(0,str(METRICS))
    from measure import official
    metric=official()
    with np.load(DATA) as z:
        gold=z['y']
    tables={}
    counts=np.zeros(10,int)
    for arm,values in {**base,**out}.items():
        M=np.empty((8,10))
        for i,o in enumerate(ORDERS):
            for j,p in enumerate(PEOPLE):
                mask=meta['person']==p
                item=metric(gold[mask],values[i,mask],meta,mask)
                counts[j]=item['n_cells']
                M[i,j]=100*item['balanced_accuracy_cell_mean']
        tables[arm]=M
    absolute={a:summary(v,counts)['pp'] for a,v in tables.items()}
    assert abs(absolute['JS']-76.55173151211109)<1e-7 and abs(absolute['frozen']-76.27933502145761)<1e-7
    pairs=[]
    for seed in SEEDS:
        for f in ('whole','half'):
            k=f+'|'+str(seed)+'|'
            pairs.extend([(k+a,k+b) for a,b in [('D','C'),('D','P'),('A','D'),('A','donor')]])
            pairs.extend([(k+'A',b) for b in ('frozen','JS')])
        for arm in ARMS:
            pairs.append((f'half|{seed}|{arm}',f'whole|{seed}|{arm}'))
    diff={a+' minus '+b:summary(tables[a]-tables[b],counts) for a,b in pairs}
    Path('result.json').write_text(json.dumps(dict(before=sha(before),scores=absolute,contrasts=diff),indent=2))
    print(absolute,flush=True)


if __name__=='__main__':
    main()
