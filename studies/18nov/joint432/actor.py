"""Restore fitted A6 and verify unchanged native and residual identity on each shuffled stream."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

WORK=Path('/mnt/lustre/work/gehler/gfh527')
sys.path[:0]=[str(WORK/'ni26_joint382_v1_20261118'),str(WORK/'ni26_residual398_v1_20261118'),
              str(WORK/'ni26_conflict388_v1_20261118')]
from adapter import NativeLayer, Residual
from restore import init, joint, prior
from actor388 import ARCH, ROWS, SEEDS
ARMS=('R','K','bypass','A6','J_real','frozen')


def lock():
    records={}
    for seed in SEEDS:
        for arm in ('R','K'):
            path=Path('fits')/f'{arm}_{seed}.json'
            r=json.loads(path.read_text())
            assert (r['arm'],r['seed'],r['steps'],r['rows'])==(arm,seed,120,ROWS)
            joint.pin(path.with_suffix('.pt'),r['weights'])
            joint.pin(Path('fits/source.sha256'),r['source'])
            records[f'{arm}_{seed}']=r
        a,b=(records[f'{k}_{seed}'] for k in ('R','K'))
        assert a['episodes']==b['episodes'] and a['fixed']==b['fixed']
        assert all(a['before'][k]==v for k,v in b['before'].items())
    return records


def model(native,arm,seed):
    n=init(native,'K')
    if arm!='zero':
        k='R' if arm=='bypass' else arm
        ck=torch.load(f'fits/{k}_{seed}.pt',map_location='cpu',weights_only=True)
        n.load_state_dict(ck['native'])
    if arm in ('zero','R'):
        r=Residual(True)
        if arm=='R':
            r.load_state_dict(ck['residual'])
        n.spatial_conv[1]=NativeLayer(n.spatial_conv[1],r)
    return n.eval()


@torch.no_grad()
def actor(person):
    assert json.loads(Path('fits.json').read_text())==lock()
    d,people,rows,runtime=prior.source()
    prior.owners(rows)
    streams,held,_,people2=joint.banks(joint.verify())
    test=held[person]
    assert joint.digest(rows)==ROWS and np.array_equal(people,people2)
    assert np.array_equal(test,np.flatnonzero(people==person)) and not np.intersect1d(rows,test).size
    assert len(set(d['stream'][test]))==1
    perms=joint.permutations(len(people))
    zero=model(runtime.net,'zero',382)
    base=init(runtime.net,'K')
    bank={}
    for seed,(root,digest) in ARCH.items():
        joint.pin(root/'before_labels.json',digest)
        proof=json.loads((root/'before_labels.json').read_text())['actors'][person]
        joint.pin(root/(person+'.npz'),proof['outputs_sha256'])
        assert proof['train_rows']==ROWS and proof['test_rows']==joint.digest(test)
        nets={a:model(runtime.net,a,seed) for a in ('R','bypass','K')}
        with np.load(root/(person+'.npz')) as z:
            assert np.array_equal(z['index'],test)
            for i,o in enumerate(joint.ORDERS):
                pos=streams[person][o][2]
                rr=test[pos]
                assert np.array_equal(rr,perms[o][np.isin(perms[o],test)])
                undo=np.argsort(pos)
                x=torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
                gold=base(x)
                assert torch.equal(zero(x),gold)
                prob=gold.double().softmax(1).numpy()[undo]
                ref=z['A6|'+o]
                assert np.max(abs(prob-ref))<1e-6 and np.array_equal(prob.argmax(1),ref.argmax(1))
                for a in ('A6','J_real','frozen'):
                    bank[f'{seed}|{a}|{o}']=z[f'{a}|{o}'].argmax(1).astype(np.int8)
                for a,n in nets.items():
                    out=n(x)
                    assert torch.isfinite(out).all()
                    if i==0:
                        torch.testing.assert_close(n(x[:8]),out[:8],atol=3e-4,rtol=1e-5)
                    bank[f'{seed}|{a}|{o}']=out.argmax(1).numpy()[undo].astype(np.int8)
    dest=Path(person+'.npz')
    np.savez_compressed(dest,index=test,**bank)
    record=dict(held=person,train=ROWS,test=joint.digest(test),output=joint.sha(dest),
                fits=joint.sha(Path('fits.json')),source=joint.sha(Path('source.sha256')))
    Path(person+'.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='lock':
        Path('fits.json').write_text(json.dumps(lock(),indent=2))
    else:
        actor(sys.argv[1])
