"""Apply one source-fit native waveform model at a time to each held session."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from lock_fits import PRIMARY, SOURCE, lock, sha
sys.path.insert(0,str(PRIMARY))
from restore import PRIOR, init, joint, prior

PRIOR_PROOF='5f16bcf878cd11b0a4b2cb8ca5b4871aa7f57a7d4781dda54fb25a6616c71a30'
BASE=('population','A6','frozen','JS','C_real','D_real','DC_real')
ARMS=BASE+tuple(a+'_'+m for a in 'FJ' for m in ('real','blank','donor'))+('K',)


def verify(root,seed,expected):
    joint.pin(Path('fit_lock.json'),expected)
    assert lock(root,seed)==json.loads(Path('fit_lock.json').read_text())
    joint.check_manifest(PRIMARY,SOURCE)
    joint.pin(PRIOR/'before_labels.json',PRIOR_PROOF)


@torch.no_grad()
def run(person,root,seed,expected):
    start=time.monotonic()
    verify(root,seed,expected)
    d,people,rows,runtime=prior.source()
    prior.owners(rows)
    streams,held,_,people2=joint.banks(joint.verify())
    test=held[person]
    assert np.array_equal(people,people2) and np.array_equal(test,np.flatnonzero(people==person))
    assert not np.intersect1d(test,rows).size and len(set(d['stream'][test]))==1
    donor=np.flatnonzero((people=='P04')&(d['session'].astype(str)=='1'))
    assert len(donor)==189 and np.isin(donor,rows).all()
    models={a:init(runtime.net,a) for a in 'FJK'}
    for a,model in models.items():
        model.load_state_dict(torch.load(root/f'{a}_{seed}.pt',weights_only=True,map_location='cpu'))
    donor_x=torch.from_numpy(runtime._aligned(donor,True))
    codes={}
    for a in 'FJ':
        layer=models[a].spatial_conv[1]
        codes[a]=layer.encode(layer.standardized(prior.preactivation(models[a],donor_x)))
    arch=PRIOR/(person+'.npz')
    joint.pin(arch,json.loads((PRIOR/'before_labels.json').read_text())['actors'][person]['outputs_sha256'])
    checks={'baseline':0.,'step':0.,'prefix':0.}
    out={}
    perms=joint.permutations(len(people))
    with np.load(arch) as z:
        assert np.array_equal(test,z['index'])
        for order in joint.ORDERS:
            pos=streams[person][order][2]
            rr=test[pos]
            assert np.array_equal(rr,perms[order][np.isin(perms[order],test)])
            x=torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
            undo=np.argsort(pos)
            for arm in BASE:
                key=arm+'|'+order
                out[key]=z[key]
                if arm in ('A6','population'):
                    runtime.net.spatial_conv[1].k=20. if arm=='A6' else None
                    probs=runtime.net(x).double().softmax(1).numpy()[undo]
                    err=float(abs(probs-out[key]).max())
                    assert err<=1e-6 and np.array_equal(probs.argmax(1),out[key].argmax(1))
                    checks['baseline']=max(err,checks['baseline'])
            for a,model in models.items():
                layer=model.spatial_conv[1]
                if a!='K':
                    ind=np.searchsorted(donor,perms[order][np.isin(perms[order],donor)])
                    layer.donor=codes[a][ind]
                for mode in (('real',) if a=='K' else ('real','blank','donor')):
                    if a!='K':
                        layer.mode=mode
                    logits=model(x)
                    assert torch.isfinite(logits).all()
                    key=('K' if a=='K' else a+'_'+mode)+'|'+order
                    out[key]=logits.double().softmax(1).numpy()[undo]
                    if order==joint.ORDERS[0]:
                        short=x[:8]
                        s=model(short)
                        torch.testing.assert_close(s,logits[:8],atol=3e-4,rtol=1e-5)
                        assert torch.equal(s,model(short))
                        checks['prefix']=max(checks['prefix'],float((s-logits[:8]).abs().max()))
                        if a!='K':
                            h=prior.preactivation(model,short)
                            manual=prior.single_steps(layer,layer.standardized(h),mode,layer.donor)
                            val=layer(h)
                            torch.testing.assert_close(val,manual,atol=2e-4,rtol=1e-5)
                            checks['step']=max(checks['step'],float((val-manual).abs().max()))
    assert len(out)==len(ARMS)*8
    np.savez_compressed(person+'.npz',index=test,**out)
    record=dict(held=person,train_rows=joint.digest(rows),test_rows=joint.digest(test),seed=seed,
                train_people=joint.FIT_PEOPLE,fit_lock=expected,actor_source=sha(Path('source.sha256')),
                outputs_sha256=sha(Path(person+'.npz')),checks=checks,held_labels_opened=False,
                seconds=time.monotonic()-start)
    Path(person+'.json').write_text(json.dumps(record,indent=2))
    print(record,flush=True)


if __name__=='__main__':
    run(sys.argv[1],Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
