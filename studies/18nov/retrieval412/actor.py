"""Commit causal held-window corrections and native whole-decoder outputs."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from model import Metric, correction, features

root = Path('/mnt/lustre/work/gehler/gfh527')
parent = root/'ni26_joint382_v1_20261118'
sys.path.insert(0,str(parent))
from restore import joint, prior

ARCH = root/'ni26_joint382_actor_v1_20261118'
SEEDS = (412,1412)
PEOPLE = ('P01','P02','P03','P09')
ARMS = ('real','destroyed','cyclic','donor')


def lock():
    records = {}
    for seed in SEEDS:
        for arm in ARMS[:2]:
            path = Path('fits')/f'{arm}_{seed}.json'
            rec = json.loads(path.read_text())
            assert rec['steps']==120 and rec['seed']==seed and rec['arm']==arm and 'H100' in rec['device']
            assert rec['source']=='5ce8870776212d97df48c630e7fffec22482b6141224a0aaf5585987be43b800'
            assert rec['rows']=='882b5b97ec811d697540f6152be0d02eb49448e9e4c88c3d6c28fcb1433b2f1f'
            joint.pin(path.with_suffix('.pt'),rec['weights'])
            records[f'{seed}_{arm}']=rec
        a,b=(records[f'{seed}_{arm}'] for arm in ARMS[:2])
        assert a['schedule']==b['schedule'] and a['before']==b['before'] and a['final']!=b['final']
    assert records['412_real']['schedule']!=records['1412_real']['schedule']
    return records


def embed(raw, weights):
    net=Metric(raw.shape[1])
    net.project.weight.copy_(weights['weight'])
    return net((raw-weights['mean'])/weights['scale']).double()


@torch.no_grad()
def run(person, smoke):
    assert lock()==json.loads(Path('fits.json').read_text())
    joint.check_manifest(parent,'67829e9fe023605f5426d1bb035f9b51f00e8fc026c9b6824fdae3035b92c5a9')
    d,people,source,runtime=prior.source()
    prior.owners(source)
    assert runtime.align is False
    streams,held,_,people2=joint.banks(joint.verify())
    test=held[person]
    assert np.array_equal(people,people2) and np.array_equal(test,np.flatnonzero(people==person))
    assert not np.intersect1d(source,test).size and len(set(d['stream'][test]))==1
    donor=np.flatnonzero((people=='P04') & (d['session'].astype(str)=='1'))
    assert len(donor)==189 and np.isin(donor,source).all()
    joint.pin(ARCH/'before_labels.json','2c1e872be6575e154ee59120f36d4c7085c1a3c1619efbb70abb18a56c6e9e77')
    archive=json.loads((ARCH/'before_labels.json').read_text())['actors'][person]
    joint.pin(ARCH/(person+'.npz'),archive['outputs_sha256'])
    perms=joint.permutations(len(people))
    net=runtime.net
    raw=lambda rr: torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
    net.spatial_conv[1].k=None
    donor_x=raw(donor)
    donor_feat=features(net,donor_x)
    donor_votes=net(donor_x).double().softmax(1)
    out={}
    with np.load(ARCH/(person+'.npz')) as z:
        assert np.array_equal(z['index'],test)
        for o in joint.ORDERS[:1] if smoke else joint.ORDERS:
            pos=streams[person][o][2]
            rr=test[pos]
            assert np.array_equal(rr,perms[o][np.isin(perms[o],test)])
            dp=np.searchsorted(donor,perms[o][np.isin(perms[o],donor)])
            dp=np.resize(dp,len(rr))
            x=raw(rr)
            feat=features(net,x)
            net.spatial_conv[1].k=None
            votes=net(x).double().softmax(1)
            net.spatial_conv[1].k=20.
            logits=net(x).double()
            for a,v in [('population',votes),('A6',logits.softmax(1))]:
                ref=z[a+'|'+o][pos]
                assert np.max(abs(v.numpy()-ref))<=1e-6 and np.array_equal(v.argmax(1).numpy(),ref.argmax(1))
            undo=np.argsort(pos)
            out['native|'+o]=logits.numpy()[undo]
            for seed in SEEDS:
                for arm in ARMS[:2]:
                    w=torch.load(Path('fits')/f'{arm}_{seed}.pt',map_location='cpu',weights_only=True)
                    q=embed(feat,w)
                    corrections={arm:correction(q,q,votes)}
                    if arm=='real':
                        corrections['cyclic']=correction(q,q,votes,True)
                        corrections['donor']=correction(q,embed(donor_feat,w)[dp],donor_votes[dp])
                        c=corrections[arm]
                        torch.testing.assert_close(c[:15],correction(q[:15],q[:15],votes[:15]))
                        order=torch.tensor([*reversed(range(14)),14])
                        qr=embed(features(net,x[order]),w)
                        net.spatial_conv[1].k=None
                        vr=net(x[order]).double().softmax(1)
                        torch.testing.assert_close(c[14],correction(qr,qr,vr)[14],atol=1e-6,rtol=0)
                    for name,delta in corrections.items():
                        assert not delta[0].any() and delta.abs().max()>.001
                        out[f'{seed}|{name}|{o}']=delta.numpy()[undo]
    dest=Path(('smoke_' if smoke else '')+person+'.npz')
    np.savez_compressed(dest,index=test,**out)
    dest.with_suffix('.json').write_text(json.dumps(dict(source=joint.sha(Path('source.sha256')),
        fits=joint.sha(Path('fits.json')),outputs=joint.sha(dest),held=person,labels_opened=False)))


if __name__=='__main__':
    if sys.argv[1]=='lock':
        with open('fits.json','x') as f:
            json.dump(lock(),f,indent=2)
    else:
        run(sys.argv[1],sys.argv[2:]==['smoke'])
