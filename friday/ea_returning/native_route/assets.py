"""Authenticate the matching source-fold and returning-route artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT=Path('/assets')
OWNER=ROOT/'ni26_aux431_20261118'
TAPE=ROOT/'ni26_friday_returning_tail_v2_20261118'
BRIDGE=ROOT/'ni26_friday_ea_returning_bridge_v2_20261118'
BASE='ni26_js321_20261117_returning_'
PREFIXES=('fresh_recording|','fresh_causal|')
VISITS=('P04|2','P04|3','P10|3','P10|2')
ANCHORS=dict(zip(VISITS,(2153,2383,1326,1238)))
ZIP='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def digest(rows):
    return hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest()


def meta(path):
    return json.loads(Path(path).read_text())


def package():
    path=Path('/package/submission.py')
    spec=importlib.util.spec_from_file_location('pinned_routeb',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    assert module.RouteB.stepx.__globals__['__file__']==str(path)
    with np.load('/package/weights.npz',allow_pickle=False) as f:
        z={k:f[k] for k in f.files}
    assert z['tsem_V'].shape==(44239,128) and z['route_V_hi'].shape==(44239,128)
    assert int(z['v6_tpri_on'])==int(z['v6_routed'])==1
    assert float(z['v6_tpri_kappa'])==20. and float(z['own_beta'])==.10
    return module,z


def source_fit(who,people,days):
    fold=who+'_s23'; p=0 if who=='P04' else 1
    expected=(2278,377,189) if p==0 else (2270,385,192)
    subject=people=='Dreyer2026Proteus/'+who
    mask=subject & np.isin(days,('2','3')); train,held=np.flatnonzero(~mask),np.flatnonzero(mask)
    own=subject & (days=='1')
    assert (len(train),len(held),int(own.sum()))==expected and not (own & mask).any()
    assert np.isin(np.flatnonzero(own),train).all()
    fit=ROOT/(BASE+('view_fit1' if p==0 else 'view_fit_p10_1'))
    view=ROOT/(BASE+'view_bank1')/fold
    contexts=ROOT/(BASE+'context_new1')/f'context_{p}'
    source=fit/f'source_view_{fold}.npz'; views=view/'full0.npz'; context=contexts/f'context_{fold}.npz'
    fit_meta,bank_meta,context_meta=(meta(x) for x in (fit/'fit.json',view/'proof.json',context.with_suffix('.json')))
    source_sha=sha(source)
    assert fit_meta['status']=='original_source_core_PCA256_fit_complete_NOT_raw_parity'
    assert bank_meta['status']=='full_native_raw_causal256_eightorder_bank_passed_NOT_accuracy'
    assert context_meta['status']=='fold_personal_context_source_fit_complete_ENGINEERING_ONLY'
    assert fit_meta['artifact_sha256']==bank_meta['source_fit_sha256']==context_meta['feature_basis_sha256']==source_sha
    assert bank_meta['files']['full0.npz']==sha(views) and context_meta['arrays_sha256']==sha(context)
    assert context_meta['frozen_zip_sha256']==ZIP and bank_meta['executor']=='actual_clean_worker_SIF_CPU'
    for proof,flag in ((fit_meta,'query_labels_used'),(bank_meta,'query_labels_read'),(context_meta,'query_labels_used')):
        assert proof['fold']==fold and not proof[flag] and not proof['accuracy_computed']
        assert proof['train_rows_sha256']==digest(train) and proof['held_rows_sha256']==digest(held)
    assert context_meta['person_source_hashes'][who]==digest(np.flatnonzero(own))
    with np.load(source,allow_pickle=False) as f:
        assert np.array_equal(f['idx_tr'],train)
        core={k[3:]:f[k] for k in f.files if k.startswith('ts_')}
        m,low,high=(f[k].astype(np.float64) for k in ('tsem_m','tsem_V','route_V_hi'))
    with np.load(views,allow_pickle=False) as f:
        assert np.array_equal(f['index'],held)
        reference={k:f[k] for k in ('proba','view256')}
    with np.load(context,allow_pickle=False) as f:
        params={k:f[k] for k in f.files if k!='metadata'}
    assert (m.shape,low.shape,high.shape)==((44239,),(44239,128),(44239,128))
    assert any(v['order']=='full0' for v in bank_meta['checks'])
    return dict(fold=fold,fit_sha=source_sha,context_sha=sha(context),view_sha=sha(views),held=held,
                params=params,core=core,m=m,low=low,high=high,reference=reference)


def load_rows():
    for stem,path in (('zero',TAPE),('matched',TAPE),('matched',BRIDGE)):
        j=meta(path/f'{stem}.json')
        assert sha(path/f'{stem}.npz')==j['bank']
        if path==TAPE:
            assert j['status']=='passed_label_blind_returning_tail' and j['exact_reverse_reset']
            assert 'no gold' in j['scope'] and j['source']=='bb4c820bf8c180faf3321560facc5df3b38fa299896163d45734bfd9d110b40b'
        else:
            assert j['status']=='passed' and not j['scored'] and j['label_free_targets']
    with np.load(OWNER/'data/proteus_windows.npz',allow_pickle=False) as f:
        x,p,d=(f[k] for k in ('X','subject','session'))
    assert x.shape==(2655,41,480) and x.dtype==np.float32 and np.isfinite(x).all()
    with np.load(OWNER/'runs/shuffle_perms.npz',allow_pickle=False) as f:
        order=f['full|0|seen_session']
    idx=np.sort(order); members=[]
    for i in range(3):
        with np.load(OWNER/f'runs/ocushuf_ss/full0/oc_nomix-seen_session-s{i}.npz',allow_pickle=False) as f:
            assert np.array_equal(f['index'],idx) and str(f['order_sha'])==digest(order)
            members.append(f['proba'])
    eyes=np.log(np.clip(np.mean(members,0),1e-3,1))
    assert eyes.shape==(762,3)
    return x,p,d,order,eyes


def visit_case(t,old,x,p,d,order,eyes,name,prefix):
    who,day=name.split('|'); tag=prefix+name+'|'
    rows=order[(p[order]=='Dreyer2026Proteus/'+who)&(d[order]==day)]
    idx=t['index']; i=np.searchsorted(idx,rows)
    assert np.array_equal(idx[i],rows) and rows[0]==ANCHORS[name] and np.array_equal(rows,t[tag+'rows'])
    assert np.array_equal(idx,old['index']) and np.array_equal(t['delivery'],old['delivery'])
    fields=('names','contrast_r','contrast_j','contrast_score','bio_r','bio_j','bio_score','winner_r','winner_j',
            'recognised','owner','pc256','source_head_w','source_head_c','mix','core','mixr','own_logs')
    case={k:t[tag+k] for k in fields}
    case.update({k:t[prefix+k][i] for k in ('personal_prior','kappa','tsem_pc','preown','staged','em')})
    case.update(raw=x[rows],rows=rows,eyes=eyes[i],ge=old[prefix+'GE'][i])
    assert case['ge'].shape==(len(rows),48) and str(case['owner'])==who
    assert np.array_equal(case['staged'],old[prefix+'T'][i])
    return case
