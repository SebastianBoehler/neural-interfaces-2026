"""Restore native or learned bodies from source-only H100 fits on historical CPU."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
WORK=Path('/mnt/lustre/work/gehler/gfh527')
PRIOR=WORK/'ni26_conflict388_v1_20261118'
sys.path.insert(0,str(PRIOR))
from actor388 import ARCH,PRIMARY,REFERENCES,ROWS,SEEDS,init,joint,prior

SOURCE='8be310f01637857c9d625e7df613cb7466ca2999fe2a2ee29b3f3cf46bf7d3ac'
ARMS=('JT','JP','KT','KP','JU','KU')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fits():
    joint.pin(PRIOR/'source.sha256','abf6ed121558cfae62505dc268c07edd4e9d2c21b9e4332f471cb5ffec8ccf84')
    for line in (PRIOR/'source.sha256').read_text().splitlines():
        h,f=line.split(maxsplit=1)
        joint.pin(PRIOR/f,h)
    rec={}
    for s in SEEDS:
        for a in ARMS:
            if s==1382 and a.endswith('U'):
                continue
            n=f'{a}_{s}'
            path=Path('fits')/(n+'.json')
            r=json.loads(path.read_text())
            assert r['arm']==a and r['seed']==s and r['source']==SOURCE and r['rows']==ROWS
            assert r['held_labels_opened'] is False and r['steps']==120 and 'H100' in r['device']
            assert r['bundle']=='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
            assert sha(path.with_suffix('.pt'))==r['weights']
            assert r['active']==(0 if a.endswith('U') else 119)
            if r['active']:
                assert abs(r['ratio_mean']-.35)<1e-6
            rec[n]=r
        runs=[r for r in rec.values() if r['seed']==s]
        assert len({r['episodes'] for r in runs})==1
        for body in ('J','K'):
            group=[r for r in runs if r['arm'].startswith(body)]
            for key in ('first','before','fixed'):
                assert all(r[key]==group[0][key] for r in group)
    return rec


@torch.no_grad()
def run(person):
    records=fits()
    assert json.loads(Path('fits.json').read_text())==records
    joint.check_manifest(PRIMARY,'67829e9fe023605f5426d1bb035f9b51f00e8fc026c9b6824fdae3035b92c5a9')
    d,people,rows,runtime=prior.source()
    prior.owners(rows)
    streams,held,_,people2=joint.banks(joint.verify())
    test=held[person]
    assert np.array_equal(people,people2) and joint.digest(rows)==ROWS
    assert np.array_equal(test,np.flatnonzero(people==person)) and not np.intersect1d(test,rows).size
    perms=joint.permutations(len(people))
    out,probs={},{}
    for s,(root,h) in ARCH.items():
        joint.pin(root/'before_labels.json',h)
        r=json.loads((root/'before_labels.json').read_text())['actors'][person]
        assert r['test_rows']==joint.digest(test) and r['train_rows']==ROWS
        joint.pin(root/(person+'.npz'),r['outputs_sha256'])
        models={}
        for a in ARMS:
            if s==1382 and a.endswith('U'):
                continue
            model=init(runtime.net,a[0])
            model.load_state_dict(torch.load(Path('fits')/f'{a}_{s}.pt',weights_only=True,map_location='cpu'))
            models[a]=model
        with np.load(root/(person+'.npz')) as z:
            assert np.array_equal(z['index'],test)
            for oi,o in enumerate(joint.ORDERS):
                pos=streams[person][o][2]
                rr=test[pos]
                assert np.array_equal(rr,perms[o][np.isin(perms[o],test)])
                x=torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
                undo=np.argsort(pos)
                for label,k in [('population',None),('A6',20.)]:
                    runtime.net.spatial_conv[1].k=k
                    p=runtime.net(x).double().softmax(1).numpy()[undo]
                    ref=z[label+'|'+o]
                    assert np.max(abs(p-ref))<=1e-6 and np.array_equal(p.argmax(1),ref.argmax(1))
                for label in REFERENCES:
                    out[f'{s}|{label}|{o}']=z[label+'|'+o].argmax(1).astype('int8')
                for a in ARMS:
                    if a in models:
                        net=models[a]
                        v=net(x)
                        assert torch.isfinite(v).all()
                        if oi==0:
                            torch.testing.assert_close(net(x[:8]),v[:8],atol=3e-4,rtol=1e-5)
                            assert torch.equal(net(x[:8]),net(x[:8]))
                        p=v.double().softmax(1).numpy()[undo]
                    else:
                        p=z[('J_real' if a=='JU' else 'K')+'|'+o]
                    key=f'{s}|{a}|{o}'
                    probs[key]=p
                    out[key]=p.argmax(1).astype('int8')
    np.savez_compressed(person+'.npz',index=test,**out)
    np.savez_compressed(person+'_prob.npz',index=test,**probs)
    Path(person+'.json').write_text(json.dumps(dict(held=person,train_rows=ROWS,test_rows=joint.digest(test),
        source=sha('source.sha256'),fits=sha('fits.json'),outputs=sha(person+'.npz'),probs=sha(person+'_prob.npz'),held_labels_opened=False)))


if __name__=='__main__':
    if sys.argv[1]=='lock':
        with open('fits.json','x') as f:
            json.dump(fits(),f,indent=2)
    else:
        run(sys.argv[1])
