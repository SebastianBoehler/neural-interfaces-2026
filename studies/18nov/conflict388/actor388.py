"""Run restored original A6 bodies and all six H100 checkpoints in historical CPU precision."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from verify388 import FITS,SEEDS,ROWS,verify,sha

WORK=Path('/mnt/lustre/work/gehler/gfh527')
PRIMARY=WORK/'ni26_joint382_v1_20261118'
sys.path.insert(0,str(PRIMARY))
from restore import init,joint,prior
ARCH={
 382:(WORK/'ni26_joint382_actor_v1_20261118','2c1e872be6575e154ee59120f36d4c7085c1a3c1619efbb70abb18a56c6e9e77'),
 1382:(WORK/'ni26_joint382_h100_1382_20261118/actor','0fdd714798bbd0cf7133f7e9002daa51869505f54bb673cb8c3b6e3a1be3efbb'),
}
REFERENCES=('population','A6','frozen','JS','F_real','J_real','K')


@torch.no_grad()
def run(person):
    start=time.monotonic()
    assert json.loads(Path('fits.json').read_text())=={str(k):v for k,v in verify().items()}
    joint.check_manifest(PRIMARY,'67829e9fe023605f5426d1bb035f9b51f00e8fc026c9b6824fdae3035b92c5a9')
    d,people,rows,runtime=prior.source()
    prior.owners(rows)
    streams,held,_,people2=joint.banks(joint.verify())
    test=held[person]
    assert np.array_equal(people,people2) and joint.digest(rows)==ROWS
    assert np.array_equal(test,np.flatnonzero(people==person)) and not np.intersect1d(test,rows).size
    assert len(set(d['stream'][test]))==1
    outputs={}
    models={}
    checks=dict(control=0.,prefix=0.,one_step=0.)
    for seed,(root,hash_) in ARCH.items():
        joint.pin(root/'before_labels.json',hash_)
        record=json.loads((root/'before_labels.json').read_text())['actors'][person]
        assert record['test_rows']==joint.digest(test) and record['train_rows']==ROWS
        joint.pin(root/(person+'.npz'),record['outputs_sha256'])
        with np.load(root/(person+'.npz')) as z:
            assert np.array_equal(z['index'],test)
            for arm in REFERENCES:
                for order in joint.ORDERS:
                    outputs[f'{seed}|{arm}|{order}']=z[arm+'|'+order].argmax(1).astype(np.int8)
        for arm in 'NHX':
            net=init(runtime.net,'J')
            net.load_state_dict(torch.load(FITS/f'{arm}_{seed}.pt',weights_only=True,map_location='cpu'))
            models[seed,arm]=net
    perms=joint.permutations(len(people))
    root,_=ARCH[382]
    with np.load(root/(person+'.npz')) as ref:
        for order in joint.ORDERS:
            pos=streams[person][order][2]
            rr=test[pos]
            assert np.array_equal(rr,perms[order][np.isin(perms[order],test)])
            x=torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
            undo=np.argsort(pos)
            for arm,k in [('population',None),('A6',20.)]:
                runtime.net.spatial_conv[1].k=k
                v=runtime.net(x).double().softmax(1).numpy()[undo]
                orig=ref[arm+'|'+order]
                err=float(np.abs(orig-v).max())
                assert err<=1e-6 and np.array_equal(v.argmax(1),orig.argmax(1))
                checks['control']=max(checks['control'],err)
            for (seed,arm),net in models.items():
                full=net(x)
                assert torch.isfinite(full).all()
                outputs[f'{seed}|{arm}|{order}']=full.argmax(1).numpy()[undo].astype(np.int8)
                if order==joint.ORDERS[0]:
                    short=x[:8]
                    layer=net.spatial_conv[1]
                    h=prior.preactivation(net,short)
                    manual=prior.single_steps(layer,layer.standardized(h),'real',None)
                    batch=layer(h)
                    small=net(short)
                    assert torch.equal(small,net(short))
                    torch.testing.assert_close(manual,batch,atol=2e-4,rtol=1e-5)
                    torch.testing.assert_close(small,full[:8],atol=3e-4,rtol=1e-5)
                    checks['one_step']=max(checks['one_step'],float((manual-batch).abs().max()))
                    checks['prefix']=max(checks['prefix'],float((full[:8]-small).abs().max()))
    assert len(outputs)==2*10*8
    dest=Path(person+'.npz')
    np.savez_compressed(dest,index=test,**outputs)
    rec=dict(held=person,train_rows=ROWS,test_rows=joint.digest(test),fits=sha(Path('fits.json')),
             source=sha(Path('source.sha256')),outputs=sha(dest),checks=checks,
             seconds=time.monotonic()-start,held_labels_opened=False)
    Path(person+'.json').write_text(json.dumps(rec,indent=2))
    print(rec,flush=True)


if __name__=='__main__':
    run(sys.argv[1])
