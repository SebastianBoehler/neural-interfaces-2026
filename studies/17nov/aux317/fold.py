"""§317 CPU nested-source producer. Never computes outer accuracy."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(os.environ['NI26_ROOT'])
sys.path.insert(0,str(ROOT/'experiments'))
import protocol
from data import ORDERS,PEOPLE,probabilities,sha,shin,take
from encoder import fingerprint
from pilot import identity_guard
from train import train


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('task',type=int); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not 0<=args.task<60 or os.environ['NI26_DEVICE']!='cpu':
        raise ValueError('registered CPU task0..59 required')
    torch.set_num_threads(6)
    held=PEOPLE[args.task%10]; seed=args.task//10%2; arm='ABC'[args.task//20]
    ws=Path.cwd(); output=ws/('smoke_output' if args.smoke else 'output'); output.mkdir(exist_ok=True)
    dest=output/f'{arm}_s{seed}_{held}.npz'
    if dest.exists():
        raise FileExistsError(dest)
    d=protocol.load(); universe=sorted(set(d['subject'].tolist()))
    target=next(p for p in universe if p.endswith('/'+held))
    fit=[p for p in universe if p!=target]; val=fit[seed]; inner=[p for p in fit if p!=val]
    if len(fit)!=9 or len(inner)!=8 or target in inner:
        raise ValueError('source split invalid')
    sr,x,y,scale=take(d,inner); _,vx,vy,_=take(d,[val],scale)
    (sx,sy,ss,svx,svy),sm=shin(ROOT,d['ch_names'])
    guard=identity_guard(x[:8],y[:8],sx[:8],sy[:8],seed)
    selected=train(arm,x,y,vx,vy,sx,sy,ss,svx,svy,1 if args.smoke else 40,seed=seed)
    epochs=selected['selected_epochs']
    del x,y,vx,vy
    rows,x,y,scale=take(d,fit)
    diagnostic=np.concatenate([np.flatnonzero(y.numpy()==c)[:16] for c in (0,1,2)])
    fitted,net=train(arm,x,y,x[diagnostic],y[diagnostic],sx,sy,ss,svx,svy,epochs,
                     seed=seed,select=False,return_model=True)
    tr=np.flatnonzero(d['subject']==target)
    # Outer labels are never indexed. Inference is stateless fixed population-BN.
    tx=torch.from_numpy(d['X'][tr].astype(np.float32))/scale
    before=fingerprint(net); p=probabilities(net,tx)
    if not np.allclose(p[:17],probabilities(net,tx[:17]),atol=2e-5,rtol=2e-5):
        raise ValueError('prefix/batch invariance failed')
    probabilities(net,x[:17])
    if fingerprint(net)!=before:
        raise ValueError('inference mutated model buffers/parameters')
    arrays={}
    with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as permutations:
        for order in ORDERS:
            perm=permutations[f'{order[:-1]}|{order[-1]}|all_loso']
            if not np.array_equal(np.sort(perm),np.arange(len(d['X']))):
                raise ValueError('pinned order is not a complete permutation')
            rank=np.empty(len(perm),dtype=int); rank[perm]=np.arange(len(perm))
            idx=np.argsort(rank[tr],kind='stable'); q=probabilities(net,tx[idx]); restored=np.empty_like(q); restored[idx]=q
            if not np.allclose(restored,p,atol=2e-5,rtol=2e-5):
                raise ValueError('stateless order invariance failed')
            arrays['proba_'+order]=restored
    checkpoint=dest.with_suffix('.pt')
    torch.save(dict(state=net.state_dict(),scale=scale,seed=seed,arm=arm,selected_epochs=epochs),checkpoint)
    metadata=dict(held=held,seed=seed,arm=arm,fit_people=fit,inner_fit_people=inner,validation_person=val,
                  selected_epochs=epochs,selected=selected,refit=fitted,shin=sm,guard=guard,
                  source_manifest=sha(ws/'source.sha256'),checkpoint_sha256=sha(checkpoint),model_sha256=before,
                  data_sha256=sha(ROOT/'data/proteus_windows.npz'),permutation_sha256=sha(ROOT/'runs/shuffle_perms.npz'),
                  source_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),inner_rows_sha256=hashlib.sha256(sr.tobytes()).hexdigest(),
                  normalization='source-fit per-corpus scalar;PROTEUS-only moment skip scaler;population BN;auxiliary BN frozen',
                  held_accuracy_computed=False,orders=ORDERS,smoke=args.smoke)
    with dest.open('xb') as f:
        np.savez(f,index=tr,metadata=json.dumps(metadata),**arrays)
    print('complete source-exclusive producer',dest,flush=True)


if __name__=='__main__':
    main()
