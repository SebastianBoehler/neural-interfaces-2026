"""Produce unlabeled held-person §307 posterior streams; never score them."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
CACHE = Path(os.environ['S307_CACHE']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT/'experiments'), str(ROOT/'studies/16nov/followups/native_core')]
import protocol
from bridge import Native
from adapt import FEATURES, BLOCKS, Session
from crossfit import fit_native, sha
from ids import person_ids

ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
ARMS = ('anchor','fixed','scalar','blocks','scalar0','scalar1','blocks0','blocks1')


def meta_parameter(outer, kind, seed, device):
    path = ROOT/f'runs/s307/meta/{outer}_{kind}_s{seed}.npz'
    with np.load(path,allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'].item())); p = z['parameter'].copy()
    if (meta['outer'],meta['kind'],meta['seed'],meta['smoke'],meta['held_accuracy_computed']) != (outer,kind,seed,False,False):
        raise ValueError('meta model provenance mismatch')
    if p.shape != (() if kind == 'scalar' else (BLOCKS,)) or not np.isfinite(p).all() or meta['meta_gradient_max'] <= 0:
        raise ValueError('invalid learned multiplier')
    return torch.as_tensor(np.tanh(p),dtype=torch.float64,device=device)


def main(held):
    people = [f'P{i:02d}' for i in range(1,11)]
    if held not in people: raise ValueError('invalid held person')
    target = ROOT/f'runs/s307/target/{held}.npz'
    if target.exists(): raise FileExistsError(target)
    if sha(CACHE/'producer.sha256') != os.environ['S307_CACHE_LOCK_SHA']: raise ValueError('cache source lock differs')
    d = protocol.load(); subject = person_ids(d['subject']); sessions = d['stream']
    train_people = [p for p in people if p != held]
    features, indices, priors, counts, cache_hashes, saved = [], [], [], [], {}, None
    for person in people:
        path = CACHE/f'runs/s301_native_inputs/{person}.npz'; cache_hashes[person] = sha(path)
        with np.load(path,allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item())); rows = z['rows'].copy()
            if meta['person'] != person or meta['source_sha256'] != os.environ['S307_CACHE_LOCK_SHA'] or not np.array_equal(rows,np.flatnonzero(subject==person)):
                raise ValueError('cached person source mismatch')
            if person == held:
                saved = {'rows':rows,'cov':z['cov'].copy(),'slow':z['slow'].copy()}
            else:
                features.append(z['features'].copy()); indices.append(rows)
                priors.append(z['prior'].copy()); counts.append(len(rows))
    train_rows = np.concatenate(indices); position = np.argsort(train_rows); train_rows = train_rows[position]
    if not np.array_equal(train_rows,np.flatnonzero(subject != held)) or held in set(subject[train_rows]):
        raise ValueError('target person entered anchor fit')
    X = np.ascontiguousarray(np.concatenate(features)[position],dtype=np.float64)
    scaler = StandardScaler().fit(X)
    train = np.ascontiguousarray(scaler.transform(X),dtype=np.float64)
    labels = np.ascontiguousarray(d['y'][train_rows],dtype=np.int64)
    native = Native(); start = time.monotonic()
    theta,loss,grad,calls = fit_native(native,train,labels)
    prior = np.ascontiguousarray(np.average(np.stack(priors),axis=0,weights=counts))
    device = torch.device('cuda:0')
    anchor = torch.as_tensor(theta[:,:-1],dtype=torch.float64,device=device)
    bias = torch.as_tensor(theta[:,-1],dtype=torch.float64,device=device)
    parameters = {(a,s):meta_parameter(held,a,s,device) for a in ('scalar','blocks') for s in (0,1)}
    payload = {'rows':saved['rows']}; counts_by_order = {}; max_delta = 0.
    with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z:
        for order in ORDERS:
            perm = z[f'{order[:-1]}|{order[-1]}|all_loso'].astype(np.int64)
            if not np.array_equal(np.sort(perm),np.arange(len(subject))): raise ValueError('invalid delivery permutation')
            delivered = perm[subject[perm] == held]
            output = {a:np.full((len(delivered),3),np.nan) for a in ARMS}
            session_updates = 0
            for session in dict.fromkeys(sessions[delivered]):
                group = delivered[sessions[delivered] == session]; loc = np.searchsorted(saved['rows'],group)
                raw = native.causal(prior,np.ascontiguousarray(saved['cov'][loc]),np.ascontiguousarray(saved['slow'][loc]))
                x = torch.as_tensor((raw-scaler.mean_)/scaler.scale_,dtype=torch.float64,device=device)
                handles = {'anchor':[Session(anchor,bias,torch.tensor(0.,device=device,dtype=torch.float64),enabled=False)],
                           'zero':[Session(anchor,bias,torch.tensor(0.,device=device,dtype=torch.float64))],
                           'fixed':[Session(anchor,bias,torch.tensor(1.,device=device,dtype=torch.float64))],
                           'scalar':[Session(anchor,bias,parameters['scalar',s]) for s in (0,1)],
                           'blocks':[Session(anchor,bias,parameters['blocks',s]) for s in (0,1)]}
                fixed_prefix = []
                for j,row in enumerate(x):
                    raw_vals = {a:[h.step(row) for h in hs] for a,hs in handles.items()}
                    vals = {a:torch.stack(v).mean(0) for a,v in raw_vals.items()}
                    for a in ('scalar','blocks'):
                        for seed in (0,1): vals[f'{a}{seed}'] = raw_vals[a][seed]
                    if not torch.equal(vals['anchor'],vals['zero']): raise ValueError('zero update changed frozen prediction')
                    for a in ARMS: output[a][loc[j]] = vals[a].cpu().numpy()
                    if j < 61: fixed_prefix.append(vals['fixed'].clone())
                replay = Session(anchor,bias,torch.tensor(1.,device=device,dtype=torch.float64))
                for j,row in enumerate(x[:len(fixed_prefix)]):
                    if not torch.equal(replay.step(row),fixed_prefix[j]):
                        raise ValueError('real target prefix or reset mismatch')
                session_updates += handles['fixed'][0].updates
                for h in (v for hs in handles.values() for v in hs):
                    ratio = float(torch.linalg.vector_norm(h.head[:,:-328]-anchor[:,:-328]) / torch.linalg.vector_norm(anchor[:,:-328]))
                    if ratio > .03000001: raise ValueError('target adaptation cap exceeded')
                    max_delta = max(max_delta,ratio)
            if session_updates == 0: raise ValueError('no eligible target updates')
            counts_by_order[order] = session_updates
            for a in ARMS:
                p = output[a]
                if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0): raise ValueError('invalid target posterior')
                payload[f'{a}|{order}'] = p
    meta = {'held':held,'train_people':train_people,'cache_sha256':cache_hashes,'fit_loss':loss,'fit_gradmax':grad,'fit_calls':calls,
            'dataset_sha256':sha(ROOT/'data/proteus_windows.npz'),'permutation_sha256':sha(ROOT/'runs/shuffle_perms.npz'),
            'models_sha256':{f'{a}_s{s}':sha(ROOT/f'runs/s307/meta/{held}_{a}_s{s}.npz') for a in ('scalar','blocks') for s in (0,1)},
            'updates':counts_by_order,'max_delta_ratio':max_delta,'seconds':time.monotonic()-start,'held_accuracy_computed':False}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f: np.savez(f,**payload,metadata=json.dumps(meta,sort_keys=True))
    print(f'{held}: all held streams generated without target labels or scores; {meta["seconds"]:.1f}s',flush=True)


if __name__ == '__main__': main(sys.argv[1])
