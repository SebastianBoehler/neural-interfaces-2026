"""Reconstruct one historical seen-session Sinc fold; no treatment score.

Fixed seed 0 and within0. A fold checkpoint is written only after every held
session agrees with the pinned archived QF frozen outputs. Candidate source
revision 4370f1f is not proven to be the original remote training checkout.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CFG = 'sincshallow_ea_ptm7r_rot05_sbn'
SOURCE_REV = '4370f1f'
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
PERM_SHA = '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'
TRUNK_SHA = 'f28f2865cf666bc85d872aacea2f2fb147dfc3d3835d6bf3ff2df55aa949ab8f'
QF_SHA = '4a4fe30cfb2726ee2e87cf8adeabbb2ffd8c535d65d8ede2c8fbec7468c6e20b'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def check_fold(d, fold, train, held):
    if fold not in ('P04_s23','P10_s23'): raise ValueError('unsupported fold')
    who='Dreyer2026Proteus/'+fold.split('_')[0]
    subject, session=np.asarray(d['subject']),np.asarray(d['session'])
    train, held=np.asarray(train,dtype=bool),np.asarray(held,dtype=bool)
    expected=(subject==who)&(session!='1'); own=(subject==who)&(session=='1')
    if (train.shape!=held.shape or train.shape!=subject.shape or not own.any() or
        not np.array_equal(held,expected) or not np.array_equal(train,~expected) or
        set(session[held])!={'2','3'} or not train[own].all()):
        raise ValueError(f'fold leakage or wrong train/session masks: {fold}')


def check_parity(actual, reference):
    actual, reference=np.asarray(actual),np.asarray(reference)
    if actual.shape!=reference.shape or actual.ndim!=2 or actual.shape[1]!=3 or not len(actual):
        raise ValueError('parity shape mismatch')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('parity arrays must be finite')
    delta=float(np.max(np.abs(actual-reference)))
    labels=int(np.count_nonzero(actual.argmax(1)!=reference.argmax(1)))
    if delta>1e-5 or labels:
        raise ValueError(f'frozen parity failed: max_abs={delta:.9g}, changed_labels={labels}')
    return {'max_abs':delta,'changed_labels':labels}


def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ('P04_s23','P10_s23'):
        raise SystemExit('supply P04_s23 or P10_s23')
    fold=sys.argv[1]; seed=0
    inputs={'data':(ROOT/'data/proteus_windows.npz',DATA_SHA),
            'permutation':(ROOT/'runs/shuffle_perms.npz',PERM_SHA),
            'trunk':(ROOT/'runs/pretrain/sincshallow_ea-m7r-s0.pt',TRUNK_SHA),
            'qf':(ROOT/'runs/qf_reference/ttt18within0QF_sinc_base-seen_session-s0.npz',QF_SHA)}
    for name,(path,expected) in inputs.items():
        if not path.is_file() or digest(path)!=expected: raise ValueError(f'{name}: missing or changed input')
    out=ROOT/'runs/sinc_qf_rebuild'/f'sinc-{fold}-s0.pt'; meta=out.with_suffix('.json')
    if out.exists() or meta.exists(): raise FileExistsError('refusing to overwrite reconstruction output')
    sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'scripts')]
    import protocol
    import session_bn as SB
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('this diagnostic requires the queued Galvani GPU')
    torch.set_num_threads(8)
    d=protocol.load(); folds={name:(tr,te) for name,tr,te in protocol.folds(d,'seen_session')}
    train, held=folds[fold]; check_fold(d,fold,train,held)
    with np.load(inputs['qf'][0]) as z: idx,ref=z['index'].copy(),z['proba'].copy()
    with np.load(inputs['permutation'][0]) as z: delivery=z['within|0|seen_session'].copy()
    if len(idx)!=762 or len(np.unique(idx))!=762 or not np.array_equal(np.sort(delivery),idx):
        raise ValueError('QF index/delivery mismatch')
    held_rows=np.flatnonzero(held)
    if not np.isin(held_rows,idx).all(): raise ValueError('QF reference misses held rows')
    m=SB.make(CFG,d,seed)
    if (m.seed!=seed or m.arch!='sincshallow' or not m.align or m.rot!=.05 or m.group!=32 or
        m.epochs!=40 or m.lr!=1e-3 or m.wd!=1e-3 or m.batch!=64 or
        Path(m.init.format(seed=seed)).resolve()!=inputs['trunk'][0].resolve()):
        raise ValueError('unexpected historical training recipe')
    train_rows=np.flatnonzero(train)
    print(f'candidate_source={SOURCE_REV} fold={fold} seed={seed} train_rows={len(train_rows)} held_rows={len(held_rows)} device={torch.cuda.get_device_name(0)}',flush=True)
    m.fit(train_rows); m.set_k(20.0)
    if not str(m.dev).startswith('cuda') or getattr(m,'n_sbn',0)<=0: raise ValueError('expected GPU and active SessionBN')
    checks={}; seen=[]
    for session in dict.fromkeys(d['stream'][held_rows]):
        r=delivery[d['stream'][delivery]==session]
        expected_rows=held_rows[d['stream'][held_rows]==session]
        if not np.array_equal(np.sort(r),expected_rows): raise ValueError('session delivery mismatch')
        X=torch.from_numpy(np.ascontiguousarray(m._aligned(r,causal=True),dtype=np.float32)).to(m.dev)
        with torch.no_grad(): pred=torch.softmax(m.net(X)[:,:3].double(),1).cpu().numpy()
        pos=np.searchsorted(idx,r)
        if np.any(pos>=len(idx)) or not np.array_equal(idx[pos],r): raise ValueError('QF row alignment mismatch')
        checks[session]=check_parity(pred,ref[pos]); seen.extend(r.tolist())
        print(f'{session} n={len(r)} parity={checks[session]}',flush=True)
    if len(checks)!=2 or not np.array_equal(np.sort(seen),held_rows): raise ValueError('held sessions not covered exactly once')
    meta_data={'candidate_source':SOURCE_REV,'seed':seed,'fold':fold,'config':CFG,
               'input_sha256':{k:v[1] for k,v in inputs.items()},'train_count':len(train_rows),
               'held_count':len(held_rows),'train_rows_sha256':hashlib.sha256(train_rows.astype('<i8').tobytes()).hexdigest(),
               'held_rows_sha256':hashlib.sha256(held_rows.astype('<i8').tobytes()).hexdigest(),
               'session_parity':checks,'job_id':os.environ.get('SLURM_JOB_ID'),
               'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f: torch.save({'state':{k:v.detach().cpu() for k,v in m.net.state_dict().items()},'G':m.G,'scale':m.scale},f)
    meta_data['checkpoint_sha256']=digest(out)
    with meta.open('x') as f: json.dump(meta_data,f,indent=2,sort_keys=True); f.write('\n')
    print(f'FROZEN QF PARITY PASS checkpoint={out} sha256={meta_data["checkpoint_sha256"]}',flush=True)


if __name__=='__main__': main()
