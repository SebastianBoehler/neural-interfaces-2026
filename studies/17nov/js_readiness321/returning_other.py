"""NEW source-audited FB/FBL/raw weights; same weights in both paired returning arms."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from returning_train import prepare,export,sha

CONFIG={'fb':('fb_sbn_rot05','fbcnet',True,.05,(2.,5.)),
        'fbl':('fblight_ea_rot05_sbn','fblight',True,.05,(5.,)),
        'fbraw':('fbcnet_raw_sbn','fbcnet',False,0.,(20.,))}


def main(family,fold,seed):
    if family not in CONFIG:raise ValueError('registered family required')
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('A100 required')
    torch.set_num_threads(1);d,_,sb,train,held,_,_=prepare(fold,seed)
    cfg,arch,align,rot,priors=CONFIG[family];m=sb.make(cfg,d,seed)
    recipe=dict(cfg=cfg,arch=m.arch,align=m.align,rot=m.rot,epochs=m.epochs,lr=m.lr,wd=m.wd,
                batch=m.batch,group=m.group,input_prior_k=m.k,train_fir=m.train_fir,init=m.init)
    expected=dict(cfg=cfg,arch=arch,align=align,rot=rot,epochs=40,lr=.001,wd=.001,batch=64,
                  group=32,input_prior_k=5.,train_fir=False,init=None)
    if recipe!=expected:raise ValueError('documented family recipe differs')
    seen=[];aligned=m._aligned
    def observed(rows,*args,**kwargs):
        seen.append(np.asarray(rows).copy());return aligned(rows,*args,**kwargs)
    m._aligned=observed;start=time.monotonic();m.fit(train);m._aligned=aligned
    if not np.array_equal(np.sort(np.concatenate(seen)),train) or not np.array_equal(m.G,m.C[train].mean(0)):
        raise ValueError('actual fit/normalization source rows differ')
    out=Path('other')/f'{family}_{fold}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    meta=dict(family=family,fold=fold,seed=seed,recipe=recipe,fit_seconds=time.monotonic()-start,
              train_rows_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),
              held_rows_sha256=hashlib.sha256(held.astype('<i8').tobytes()).hexdigest(),train_count=len(train),held_count=len(held),
              source_sha256=sha('source.sha256'),historical_identity_claimed=False,accuracy_computed=False,
              job=os.environ.get('SLURM_JOB_ID'),scope='NEW genuine held-session weights shared by paired control/JS')
    ckpt=out/'checkpoint.pt';torch.save(dict(state={k:v.detach().cpu() for k,v in m.net.state_dict().items()},
                                          G=m.G,scale=m.scale,**meta),ckpt)
    session=d['stream'][held[0]];rows=held[d['stream'][held]==session][:31]
    inputs=np.ascontiguousarray(m._aligned(rows,causal=True),dtype=np.float32);meta['exports']={}
    for k in priors:
        path=out/f'trace_k{k:g}.pt';errors=export(m,sb,inputs,path,k)
        meta['exports'][str(k)]=dict(errors=errors,sha256=sha(path))
    meta.update(checkpoint_sha256=sha(ckpt),seconds=time.monotonic()-start,status='new_genuine_other_family_source_export_passed')
    (out/'proof.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':main(sys.argv[1],sys.argv[2],int(sys.argv[3]))
