"""Fit an original six-source FBCNet and expose native causal logits on four excluded people."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

WORK=Path('/mnt/lustre/work/gehler/gfh527')
OWNER=WORK/'ni26_calibration339_v2_20261117'
sys.path.insert(0,str(OWNER))
from inputs339 import src,logits,fingerprint

LOCK='c6f199c747dbd2c38e86165735bc86553bb53a1ed30cd348f1792ccdb17ff22f'


def digest(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def excluded(block):
    return tuple(src.PEOPLE[(block+i)%10] for i in (0,1,3,5))


def main(block,seed,smoke):
    assert 0<=block<10 and seed in (0,1) and 'A100' in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    assert src.sha(OWNER/'source.sha256')==LOCK
    for line in (OWNER/'source.sha256').read_text().splitlines():
        h,k=line.split(maxsplit=1)
        assert src.sha(OWNER/k)==h
    src.check()
    with np.load(src.data323.DATA) as z:
        d={k:z[k] for k in ('X','subject','session')}
    d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session'])
    ids=np.array([s.rsplit('/',1)[-1] for s in d['subject']])
    omitted=excluded(block)
    fit=np.flatnonzero(~np.isin(ids,omitted))
    test=np.flatnonzero(np.isin(ids,omitted))
    d['y']=np.full(len(ids),-1)
    d['y'][fit]=src.source_labels.read_training_labels(src.data323.DATA,fit,len(ids))
    assert len(set(ids[fit]))==6 and (d['y'][test]==-1).all()
    with np.load(src.PROD/'runs/shuffle_perms.npz') as z:
        orders={o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in src.ORDERS}
    m=src.model322
    m.SEED=seed
    epochs=1 if smoke else 40
    x,y,scale,mu,sd,cw,groups=m.setup(d,fit,'cuda',epochs=epochs)
    model=m.Factor('population',False,'cuda')
    initial=m.param_hash(model)
    with np.load(src.PROD/'output'/f'P01_s{seed}'/'result.npz') as z:
        assert initial==json.loads(str(z['metadata']))['initial_sha256']['P0']
    tr=m.train(model,x,y,mu,sd,cw,groups)
    assert tr['steps']==sum(map(len,groups)) and len(groups)==epochs and tr['max_gradient']>0
    state=fingerprint(model)
    assert m.param_hash(model)!=initial
    subset=test if not smoke else test[:20]
    obs,checks=logits(model,scale,d,subset,orders,'cuda')
    assert fingerprint(model)==state
    tag=('smoke_' if smoke else '')+f'b{block}_s{seed}'
    path=Path(tag+'.npz')
    np.savez_compressed(path,index=subset,**{o:obs['B|'+o] for o in orders})
    doc=dict(block=block,seed=seed,excluded=omitted,fit=digest(fit),test=digest(subset),
             fit_people=sorted(set(ids[fit])),initial=initial,trained=state,checks=checks,
             training=tr,epochs=epochs,source=src.sha(Path('source.sha256')),data=src.data323.DATA_SHA,
             outputs=src.sha(path),held_labels_opened=False)
    Path(tag+'.json').write_text(json.dumps(doc,indent=2))
    print(tag,doc['training'],flush=True)


if __name__=='__main__':
    main(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3:]==['smoke'])
