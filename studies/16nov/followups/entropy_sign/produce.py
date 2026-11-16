"""Stream §310 strong-head entropy-sign arms without reading target labels."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(os.environ['NI26_WS']).resolve(strict=True)
CACHE=Path(os.environ['S307_CACHE']).resolve(strict=True)
S305=Path(os.environ['S310_S305']).resolve(strict=True)
REF=Path(os.environ['S310_REF']).resolve(strict=True)
os.environ['NI26_ROOT']=str(ROOT)
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'studies/16nov/followups/meta_adaptation'),str(ROOT/'experiments')]
import protocol
from ids import person_ids
from cpu_crossfit import CPUReference
from crossfit import sha
from adapt import Session
from flatten import Session as FlattenSession

ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
ARCHIVES={o:('coremember.npz' if o in ('within0','within1','full0','runs0') else 'coremember_within2_within3_full1_runs1.npz') for o in ORDERS}


def main(held):
    people=[f'P{i:02d}' for i in range(1,11)]
    if held not in people: raise ValueError('invalid held person')
    target=ROOT/f'runs/s310/target/{held}.npz'
    if target.exists(): raise FileExistsError(target)
    d=protocol.load(); subject=person_ids(d['subject']); sessions=d['stream']
    ref=REF/f'{held}.npz'; shard=S305/f'runs/s305_converged_shards/{held}.npz'
    with np.load(ref,allow_pickle=False) as z:
        fit_rows=z['fit_rows'].copy(); mean=z['mean'].copy(); scale=z['scale'].copy()
    if not np.array_equal(fit_rows,np.flatnonzero(subject!=held)) or np.any(scale<=0): raise ValueError('reference fold/scaler invalid')
    with np.load(shard,allow_pickle=False) as z:
        rm=json.loads(str(z['metadata'].item())); rrows=z['rows'].copy()
        coef=z['Nconverged_coef'].copy(); bias=z['Nconverged_intercept'].copy()
        original={o:z[f'Nconverged|{o}'].copy() for o in ORDERS}
    if rm['held']!=held or rm['held_accuracy_computed'] is not False or rm['reference_sha256']!=sha(ref) or rm['max_gradient']>1e-5 or not np.array_equal(rrows,np.flatnonzero(subject==held)):
        raise ValueError('§305 anchor source/fold/convergence guard failed')
    if rm['data_sha256']!=sha(ROOT/'data/proteus_windows.npz') or rm['permutation_sha256']!=sha(ROOT/'runs/shuffle_perms.npz'):
        raise ValueError('§305 data/permutation hashes changed')
    prior,counts,hashes,saved=[],[],{},None
    for person in people:
        path=CACHE/f'runs/s301_native_inputs/{person}.npz'; hashes[f'{person}.npz']=sha(path)
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata'].item())); rows=z['rows'].copy()
            if m['person']!=person or m['source_sha256']!=os.environ['S307_CACHE_LOCK_SHA'] or not np.array_equal(rows,np.flatnonzero(subject==person)):
                raise ValueError('cached source guard failed')
            if person==held: saved={'rows':rows,'cov':z['cov'].copy(),'slow':z['slow'].copy()}
            else: prior.append(z['prior'].copy()); counts.append(len(rows))
    if hashes!=rm['input_sha256']: raise ValueError('§305 source cache hashes changed')
    prior=np.ascontiguousarray(np.average(np.stack(prior),axis=0,weights=counts))
    native=CPUReference(); anchor=torch.as_tensor(coef,dtype=torch.float64); bias=torch.as_tensor(bias,dtype=torch.float64)
    payload={'rows':rrows}; stat={}; start=time.monotonic()
    with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z:
        for order in ORDERS:
            perm=z[f'{order[:-1]}|{order[-1]}|all_loso'].astype(np.int64)
            if not np.array_equal(np.sort(perm),np.arange(len(subject))): raise ValueError('invalid delivery order')
            delivered=perm[subject[perm]==held]
            outputs={a:np.full((len(rrows),3),np.nan) for a in ('anchor','sharpen','flatten')}; updates=0
            for session in dict.fromkeys(sessions[delivered]):
                group=delivered[sessions[delivered]==session]; loc=np.searchsorted(rrows,group)
                raw=native.causal(prior,np.ascontiguousarray(saved['cov'][loc]),np.ascontiguousarray(saved['slow'][loc]))
                x=torch.as_tensor((raw-mean)/scale,dtype=torch.float64)
                fixed=torch.tensor(1.,dtype=torch.float64); zero=torch.tensor(0.,dtype=torch.float64)
                handles={'anchor':Session(anchor,bias,zero,enabled=False),'zero':Session(anchor,bias,zero),
                         'sharpen':Session(anchor,bias,fixed),'flatten':FlattenSession(anchor,bias,fixed)}
                prefix=[]
                for j,row in enumerate(x):
                    p={a:h.step(row) for a,h in handles.items()}
                    if not torch.equal(p['anchor'],p['zero']): raise ValueError('zero-update identity failed')
                    for a in outputs: outputs[a][loc[j]]=p[a].numpy()
                    if j<61: prefix.append(p['flatten'].clone())
                replay=FlattenSession(anchor,bias,fixed)
                for j,row in enumerate(x[:len(prefix)]):
                    if not torch.equal(replay.step(row),prefix[j]): raise ValueError('real-data prefix/reset failed')
                updates+=handles['flatten'].updates
                if handles['flatten'].updates!=handles['sharpen'].updates: raise ValueError('update schedules differ')
            if updates<=0: raise ValueError('no active session update')
            if sha(ROOT/'runs'/ARCHIVES[order])!=rm['archives_sha256'][ARCHIVES[order]]:
                raise ValueError('§305 historical core archive changed')
            with np.load(ROOT/'runs'/ARCHIVES[order],allow_pickle=False) as archive_file:
                archive=archive_file[f'{order}|all_loso'][rrows]
            reference=archive+original[order]; delta=float(np.max(np.abs(reference-outputs['anchor'])))
            sortedp=np.sort(reference,axis=1); non_tied=sortedp[:,-1]>sortedp[:,-2]
            labels=int(np.count_nonzero((reference.argmax(1)!=outputs['anchor'].argmax(1))&non_tied))
            if delta>1e-5 or labels: raise ValueError(f'§305 anchor parity failed: {delta=} {labels=}')
            for arm,p in outputs.items():
                if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0): raise ValueError('invalid probabilities')
                payload[f'{arm}|{order}']=p
            if np.array_equal(outputs['sharpen'],outputs['anchor']) or np.array_equal(outputs['flatten'],outputs['anchor']):
                raise ValueError('active policy changed no probabilities')
            stat[order]={'anchor_maxdiff':delta,'non_tied_changed':labels,'updates':updates}
    meta={'held':held,'cache_sha256':hashes,'reference_sha256':sha(ref),'shard_sha256':sha(shard),
          'stats':stat,'seconds':time.monotonic()-start,'held_accuracy_computed':False}
    payload['metadata']=json.dumps(meta,sort_keys=True); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f: np.savez(f,**payload)
    print(f'{held}: CPU entropy-sign streams pass anchor/reset/zero, {meta["seconds"]:.1f}s; no held score',flush=True)


if __name__=='__main__': main(sys.argv[1])
