"""Verify the identity of the residual inside the genuine causal A6 path."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from adapter import NativeLayer,Residual
ROOT=Path('/mnt/lustre/work/gehler/gfh527')
PARENT=ROOT/'ni26_anchor394_v1_20261118'
sys.path.insert(0,str(PARENT))
from actor import ARCH,PRIMARY,ROWS,SEEDS,init,joint,prior,sha

PEOPLE=('P01','P02','P03','P09')
ARMS=('RT','RP','RU','CT')
SOURCE='950b042541f4522e31bcdf2c96ebc93b42ae1871b92f20a846d1be32aaefd20c'


def lock():
    rec={}
    for s in SEEDS:
        for a in ARMS:
            n=f'{a}_{s}'
            path=Path('fits')/(n+'.json')
            r=json.loads(path.read_text())
            assert r['arm']==a and r['seed']==s and r['source']==SOURCE and r['rows']==ROWS
            assert r['bundle']=='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
            assert r['active']==(0 if a=='RU' else 119) and r['steps']==120 and r['max_logit_shift']>1e-4
            assert not r['held_labels_opened'] and sha(path.with_suffix('.pt'))==r['weights'] and 'H100' in r['device']
            rec[n]=r
        assert len({r['episodes'] for r in rec.values() if r['seed']==s})==1
    assert len({r['initial'] for r in rec.values()})==len({r['fixed'] for r in rec.values()})==1
    assert rec['RT_382']['episodes']!=rec['RT_1382']['episodes']
    return rec


def build(native,a,seed):
    net=init(native,'K')
    orig=net.spatial_conv[1]
    torch.manual_seed(398)
    res=Residual(a!='CT')
    if a!='zero':
        res.load_state_dict(torch.load(Path('fits')/f'{a}_{seed}.pt',weights_only=True,map_location='cpu'))
    net.spatial_conv[1]=NativeLayer(orig,res).eval()
    return net.eval()


@torch.no_grad()
def main(person):
    joint.pin(PARENT/'source.sha256','13ab644e8f4eaf1711c12b8f853cd7be863c6a69c6cdd37c510c56fc9983e46d')
    for line in (PARENT/'source.sha256').read_text().splitlines():
        h,f=line.split(maxsplit=1)
        joint.pin(PARENT/f,h)
    assert json.loads(Path('fits.json').read_text())==lock()
    d,people,rows,runtime=prior.source()
    prior.owners(rows)
    streams,held,_,people2=joint.banks(joint.verify())
    test=held[person]
    assert joint.digest(rows)==ROWS and np.array_equal(people,people2) and np.array_equal(test,np.flatnonzero(people==person))
    assert not np.intersect1d(rows,test).size
    root,h=ARCH[382]
    joint.pin(root/'before_labels.json',h)
    record=json.loads((root/'before_labels.json').read_text())['actors'][person]
    joint.pin(root/(person+'.npz'),record['outputs_sha256'])
    base=init(runtime.net,'K')
    zero=build(runtime.net,'zero',382)
    models={(a,s):build(runtime.net,a,s) for s in SEEDS for a in ARMS}
    perms=joint.permutations(len(people))
    bank={}
    with np.load(root/(person+'.npz')) as z:
        assert np.array_equal(z['index'],test)
        for oi,o in enumerate(joint.ORDERS):
            pos=streams[person][o][2]
            rr=test[pos]
            assert np.array_equal(rr,perms[o][np.isin(perms[o],test)])
            undo=np.argsort(pos)
            x=torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
            gold=base(x)
            assert torch.equal(zero(x),gold)
            p=gold.double().softmax(1).numpy()[undo]
            ref=z['A6|'+o]
            assert np.max(abs(p-ref))<=1e-6 and np.array_equal(p.argmax(1),ref.argmax(1))
            bank['A6|'+o]=p
            bank['JS|'+o]=z['JS|'+o]
            for (a,s),net in models.items():
                scores=net(x)
                assert torch.isfinite(scores).all() and (scores-gold).abs().max()>1e-4
                if oi==0:
                    torch.testing.assert_close(net(x[:8]),scores[:8],atol=3e-4,rtol=1e-5)
                    assert torch.equal(net(x[:8]),net(x[:8]))
                bank[f'{s}|{a}|{o}']=scores.double().softmax(1).numpy()[undo]
    np.savez_compressed(person+'.npz',index=test,**bank)
    Path(person+'.json').write_text(json.dumps(dict(source=sha('source.sha256'),fits=sha('fits.json'),
        outputs=sha(person+'.npz'),rows=joint.digest(test),held=person,held_labels_opened=False)))


if __name__=='__main__':
    if sys.argv[1]=='lock':
        with open('fits.json','x') as f:
            json.dump(lock(),f,indent=2)
    else:
        main(sys.argv[1])
