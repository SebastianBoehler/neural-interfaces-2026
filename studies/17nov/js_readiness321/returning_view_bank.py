"""No-label actual frozen raw causal256 banks for both genuine source-fold bases."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import returning_view_guard as G

ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def main(fold):
    if fold not in ('P04_s23','P10_s23'):raise ValueError('registered source fold required')
    if torch.__version__!='2.14.1+cu130' or 'neural-interfaces-2026/.venv/' in Path('/proc/self/maps').read_text():raise ValueError('actual clean worker CPU required')
    torch.set_num_threads(1);start=time.monotonic();root=G.ROOT
    native=root/'ni26_js321_20261117_returning_view_native1'/fold/'raw_view_guard.json';check=json.loads(native.read_text())
    if check['status']!='raw_causal256_reference_and_samepath_passed_NOT_accuracy' or check['executor']!='actual_clean_worker_SIF_CPU':raise ValueError('native source guard required')
    fit=G.FIT if fold=='P04_s23' else root/'ni26_js321_20261117_returning_view_fit_p10_1';file=fit/f'source_view_{fold}.npz'
    if G.sha(file)!=check['source_fit_sha256']:raise ValueError('guarded source artifact changed')
    with np.load(file) as z:
        p={k[3:]:z[k].copy() for k in z.files if k.startswith('ts_')};m=z['tsem_m'].copy();V=np.concatenate((z['tsem_V'],z['route_V_hi']),1).astype(np.float64);train=z['idx_tr'].copy()
    data=root/'neural-interfaces-2026/data/proteus_windows.npz';perms=root/'neural-interfaces-2026/runs/shuffle_perms.npz'
    if G.sha(data)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327' or G.sha(perms)!='6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f':raise ValueError('raw/order changed')
    with np.load(data,allow_pickle=True) as z:X=z['X'].copy();sub=z['subject'].copy();session=z['session'].astype(str)
    who='Dreyer2026Proteus/'+fold[:3];held=np.flatnonzero((sub==who)&(session!='1'));streams=sub+'|'+session
    if np.intersect1d(train,held).size or len(held)!=(377 if fold=='P04_s23' else 385):raise ValueError('source/held rows differ')
    cls=G.frozen_class();records=[]
    def feed(model,rows):
        out=[]
        for r in rows:
            q,z=model.step(X[r]);out.append(np.r_[q,(z-m)@V])
        if model.n!=len(rows):raise ValueError('actual causal count differs')
        return np.asarray(out)
    original={k:v.copy() for k,v in p.items()}
    with np.load(perms) as orders:
        for order in ORDERS:
            rows=orders[f'{order[:-1]}|{int(order[-1])}|seen_session'];rows=rows[sub[rows]==who]
            if not np.array_equal(np.sort(rows),held):raise ValueError('full order coverage differs')
            values=np.full((len(held),259),np.nan)
            for stream in dict.fromkeys(streams[rows]):
                r=rows[streams[rows]==stream];model=cls(p);actual=feed(model,r);model.reset_state();repeat=feed(model,r)
                same=float(np.abs(actual-repeat).max());model.reset_state();a,b=copy.deepcopy(model),copy.deepcopy(model)
                prefix=feed(a,r[:17]);feed(b,r[-17:]);tail=[]
                for idx in r[17:]:
                    q,z=a.step(X[idx]);tail.append(np.r_[q,(z-m)@V])
                independent=np.concatenate((prefix,np.asarray(tail)),0);error=float(np.abs(actual-independent).max())
                if max(same,error)>1e-8 or a.n!=len(r) or not all(np.array_equal(v,p[k]) for k,v in original.items()):raise ValueError('raw reset/prefix/deepcopy/immutable invariant failed')
                if not np.isfinite(actual).all() or not np.allclose(actual[:,:3].sum(1),1.) or np.abs(actual[:,131:]).max()<=1e-12:raise ValueError('invalid/inactive full256 bank')
                values[np.searchsorted(held,r)]=actual;records.append(dict(order=order,session=stream,count=len(r),repeat_maxabs=same,prefix_copy_maxabs=error))
            if not np.isfinite(values).all():raise ValueError('incomplete bank')
            np.savez_compressed(f'{order}.npz',index=held,proba=values[:,:3],view256=values[:,3:])
            print('raw256',fold,order,'complete',flush=True)
    proof=dict(status='full_native_raw_causal256_eightorder_bank_passed_NOT_accuracy',fold=fold,checks=records,
               source_fit_sha256=G.sha(file),native_canary_sha256=G.sha(native),source_sha256=G.sha('source.sha256'),
               train_rows_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),held_rows_sha256=hashlib.sha256(held.astype('<i8').tobytes()).hexdigest(),
               files={f'{o}.npz':G.sha(f'{o}.npz') for o in ORDERS},seconds=time.monotonic()-start,query_labels_read=False,accuracy_computed=False,
               executor='actual_clean_worker_SIF_CPU',scope='raw core/view reconstruction only; full original-inner and wholepair still required')
    Path('proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)


if __name__=='__main__':main(sys.argv[1])
