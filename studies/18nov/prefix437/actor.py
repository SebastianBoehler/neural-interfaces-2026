"""Replay each person separately, verifying both frozen endpoints on every order."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
from gate import Gate,endpoints,features,mix

WORK=Path('/mnt/lustre/work/gehler/gfh527')
sys.path[:0]=[str(WORK/'ni26_joint382_v1_20261118'),str(WORK/'ni26_conflict388_v1_20261118')]
from restore import init,joint,prior
from actor388 import ARCH, ROWS
SEEDS=(437,1437)
ARMS=('gate','constant','fixed','A6','population','donor','frozen')


def lock():
    records={}
    for seed in SEEDS:
        r=json.loads(Path(f'fits/fit_{seed}.json').read_text())
        assert r['seed']==seed and r['steps']==120 and r['rows']==ROWS
        joint.pin(Path('fits/source.sha256'),r['source'])
        records[str(seed)]=r
    return records


@torch.no_grad()
def actor(person):
    rec=lock()
    assert rec==json.loads(Path('fits.json').read_text())
    d,people,rows,runtime=prior.source()
    prior.owners(rows)
    streams,held,_,people2=joint.banks(joint.verify())
    test=held[person]
    assert joint.digest(rows)==ROWS and np.array_equal(people,people2)
    assert np.array_equal(test,np.flatnonzero(people==person)) and not np.intersect1d(rows,test).size
    assert len(set(d['stream'][test]))==1
    donor=np.flatnonzero((people=='P10')&(d['session'].astype(str)=='3'))
    assert len(donor)==199 and np.isin(donor,rows).all() and len(donor)>=len(test)
    perms=joint.permutations(len(people))
    base=init(runtime.net,'K')
    root,digest=ARCH[382]
    joint.pin(root/'before_labels.json',digest)
    proof=json.loads((root/'before_labels.json').read_text())['actors'][person]
    joint.pin(root/(person+'.npz'),proof['outputs_sha256'])
    assert proof['train_rows']==ROWS and proof['test_rows']==joint.digest(test)
    bank={}
    activity={}
    with np.load(root/(person+'.npz')) as z:
        assert np.array_equal(z['index'],test)
        for i,o in enumerate(joint.ORDERS):
            pos=streams[person][o][2]
            rr=test[pos]
            assert np.array_equal(rr,perms[o][np.isin(perms[o],test)])
            undo=np.argsort(pos)
            x=torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
            dr=perms[o][np.isin(perms[o],donor)][:len(test)]
            dx=torch.roll(torch.from_numpy(runtime._aligned(dr,True)),0,-1)
            eps=base.spatial_conv[1].eps
            k,p=endpoints(base,x,eps)
            f=features(k,p)
            df=features(*endpoints(base,dx,eps))
            assert torch.equal(f[:,0],df[:,0]) and not torch.equal(f,df)
            assert torch.equal(mix(k,p,1.),k) and torch.equal(mix(k,p,0.),p)
            for a,v in [('A6',k),('population',p)]:
                ref=z[a+'|'+o]
                prob=v.double().softmax(1).numpy()[undo]
                assert np.max(abs(prob-ref))<1e-6 and np.array_equal(prob.argmax(1),ref.argmax(1))
            for seed in SEEDS:
                g,c=[Gate(flag) for flag in (True,False)]
                for gate,state in zip((g,c),rec[str(seed)]['gates']):
                    gate.load_state_dict({n:torch.tensor(v) for n,v in state.items()})
                weights=g(f)
                assert torch.isfinite(weights).all() and (weights-g(df)).abs().max()>1e-6
                activity[f'{seed}|{o}']=[float(weights.min()),float(weights.max()),float(c(f).item())]
                logits=dict(gate=mix(k,p,weights),donor=mix(k,p,g(df)),constant=mix(k,p,c(f)),
                            fixed=mix(k,p,.9),A6=k,population=p)
                if i==0:
                    ks,ps=endpoints(base,x[:8],eps)
                    fs=features(ks,ps)
                    short=mix(ks,ps,g(fs))
                    torch.testing.assert_close(short,logits['gate'][:8],atol=3e-4,rtol=1e-5)
                    assert torch.equal(short,mix(ks,ps,g(features(ks,ps))))
                for a,v in logits.items():
                    assert torch.isfinite(v).all()
                    bank[f'{seed}|{a}|{o}']=v.argmax(1).numpy()[undo].astype(np.int8)
                bank[f'{seed}|frozen|{o}']=z['frozen|'+o].argmax(1).astype(np.int8)
    dest=Path(person+'.npz')
    np.savez_compressed(dest,index=test,**bank)
    record=dict(held=person,train=ROWS,test=joint.digest(test),output=joint.sha(dest),
                fits=joint.sha(Path('fits.json')),source=joint.sha(Path('source.sha256')),activity=activity)
    Path(person+'.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='lock':
        Path('fits.json').write_text(json.dumps(lock(),indent=2))
    else:
        actor(sys.argv[1])
