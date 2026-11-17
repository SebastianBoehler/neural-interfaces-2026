"""Apply the held-person population checkpoint on each pinned shuffled order."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from estimate import components, configure

PARENT = Path('/mnt/lustre/work/gehler/gfh527/ni26_moments332_v2_20261117')
sys.path.insert(0,str(PARENT))
import source332 as src
from views332 import prepare, SessionBN

def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def endpoint(model,x):
    with torch.no_grad():
        return model(x).double().softmax(1).cpu().numpy()


def run(held, smoke):
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name()
    assert src.sha(PARENT/'source.sha256') == '6c0f967ca5e8a7240b676de483bdc575eab2918c526fee4b832dc107d925abab'
    for row in (PARENT/'source.sha256').read_text().splitlines():
        code,name = row.split(maxsplit=1)
        assert src.sha(PARENT/name) == code
    root = Path('smoke' if smoke else held)
    root.mkdir()
    arms, records = {}, []
    for seed in ((0,) if smoke else (0,1)):
        d,fit,test,_,perms,nets,original,meta = src.load(held,seed,'cuda')
        people = np.array([s.split('/')[-1] for s in d['subject']])
        assert 'y' not in d and np.array_equal(fit,np.flatnonzero(people!=held))
        assert np.array_equal(test,np.flatnonzero(people==held))
        net,scale = nets['P']
        x = torch.tensor(d['X']/scale,dtype=torch.float32,device='cuda')
        views,_ = prepare(net,x[fit[:8]])
        variance = views['V']
        layers = [m for m in variance.body.modules() if isinstance(m, SessionBN)]
        assert len(layers)==1
        w,t = components(variance,layers[0],x[fit],d['stream'][fit])
        k = w[t>0]/t[t>0]
        scalar = float(np.exp(np.median(np.log(k))))
        one = np.ones_like(w)
        models = {a:configure(variance,SessionBN,ww,tt) for a,ww,tt in [
            ('K',w,t),('G',one*scalar,one),('P',np.roll(w,1),np.roll(t,1)),('k20',20*one,one)]}
        source = x[fit[:24]]
        k20err = np.abs(endpoint(variance,source)-endpoint(models['k20'],source)).max()
        assert k20err < 2e-5 and np.max(abs(endpoint(models['P'],source)-endpoint(models['K'],source))) > 1e-7
        for m in models.values():
            full = endpoint(m,source)
            short = endpoint(m,source[:8])
            assert np.allclose(full[:8],short,atol=2e-5,rtol=2e-5)
            endpoint(m,x[fit[-9:]])
            assert np.array_equal(short,endpoint(m,source[:8]))
        records.append(dict(seed=seed,checkpoint=meta['checkpoints']['P'],variance_ratio_k_median=scalar,
                            k_range=np.percentile(k,[0,25,75,100]).tolist(),source_sessions=len(set(d['stream'][fit])),
                            fixed_channels=int((t==0).sum()),within_sha=digest(w),between_sha=digest(t),
                            k20_source_probability_error=float(k20err)))
        if smoke:
            continue
        with np.load(PARENT/'output'/f'{held}_s{seed}.npz') as old:
            assert np.array_equal(old['index'],test)
            for order in src.ORDERS:
                perm = perms[order]
                assert np.array_equal(np.sort(perm),np.arange(len(people)))
                for a,key in [('RF','PF'),('RV','PV')]:
                    values = old[f'{key}|{order}']
                    if a=='RF':
                        assert np.array_equal(values,original[f'PF|{order}'])
                    arms.setdefault(a+'|'+order,[]).append(values)
                probs={a:np.full((len(test),3),np.nan) for a in ('P','K','G')}
                for session in np.unique(d['stream'][test]):
                    delivered = perm[np.isin(perm,test[d['stream'][test]==session])]
                    pos=np.searchsorted(test,delivered)
                    assert np.array_equal(test[pos],delivered)
                    trial=x[delivered]
                    ref=old[f'PV|{order}'][pos]
                    same=endpoint(models['k20'],trial)
                    archived = endpoint(variance,trial)
                    assert np.array_equal(ref,archived), (held, seed, order, float(np.max(abs(ref-archived))))
                    assert np.max(abs(ref-same))<5e-5 and np.array_equal(ref.argmax(1),same.argmax(1)), (held, seed, order, float(np.max(abs(ref-same))))
                    for a in probs:
                        q=endpoint(models[a],trial)
                        assert np.allclose(q[:6],endpoint(models[a],trial[:6]),atol=3e-5,rtol=2e-5)
                        probs[a][pos]=q
                for a,q in probs.items():
                    assert np.isfinite(q).all() and np.allclose(q.sum(1),1)
                    arms.setdefault(a+'|'+order,[]).append(q)
    if not smoke:
        np.savez_compressed(root/'fold.npz',index=test,**{k:np.mean(v,0) for k,v in arms.items()})
    proof=dict(held=held,labels_opened=False,seeds=[r['seed'] for r in records],records=records,
               train_people=sorted(set(people[fit])),train_rows=digest(fit),test_rows=digest(test),
               source_sha256=src.sha(Path('source.sha256')))
    if not smoke:
        proof['archive_sha256']=src.sha(root/'fold.npz')
    (root/'proof.json').write_text(json.dumps(proof,indent=2))
    print(held,records,flush=True)


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2:] == ['smoke'])
