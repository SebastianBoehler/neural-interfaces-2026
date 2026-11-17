"""Source-fold personal heads/prior/own-stage arrays for engineering injection; no held accuracy."""
import ast
import hashlib
import importlib.util
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from returning_assets import ROOT,sha

MAIN=ROOT/'neural-interfaces-2026'


def owner_functions():
    path=MAIN/'replay/rlib3.py';tree=ast.parse(path.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('stdz','ps1_heads')]
    if len(functions)!=2:raise ValueError('owning head helper anchors differ')
    ns={'np':np,'LogisticRegression':LogisticRegression};exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns)
    p=Path.cwd()/'s281_fold_stage_diag.py';tree=ast.parse(p.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arrays']
    if len(functions)!=1:raise ValueError('owning own-stage exporter absent')
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(p),'exec'),ns)
    return ns


def main(fold):
    if fold not in ('P04_s23','P10_s23'):raise ValueError('registered source fold required')
    start=time.monotonic();data=MAIN/'data/proteus_windows.npz'
    if sha(data)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327':raise ValueError('data changed')
    with np.load(data,allow_pickle=True) as z:sub=z['subject'];session=z['session'].astype(str);y=z['y'].astype(np.int64)
    held=(sub=='Dreyer2026Proteus/'+fold[:3])&(session!='1');train=np.flatnonzero(~held);query=np.flatnonzero(held)
    fit=ROOT/('ni26_js321_20261117_returning_view_fit1' if fold=='P04_s23' else 'ni26_js321_20261117_returning_view_fit_p10_1')
    file=fit/f'source_view_{fold}.npz'
    proof=json.loads((fit/'fit.json').read_text())
    if proof['status']!='original_source_core_PCA256_fit_complete_NOT_raw_parity' or proof['artifact_sha256']!=sha(file):raise ValueError('new source basis fit proof absent')
    with np.load(file,allow_pickle=False) as f:idx=f['idx_tr'].copy();pc=f['pc_tr'].astype(np.float64);labels=f['y_tr'].copy()
    if not np.array_equal(np.sort(idx),train) or not np.array_equal(labels,y[idx]) or pc.shape!=(len(train),256):raise ValueError('source feature/basis rows differ')
    stream=sub+'|'+session;per={}
    for name in dict.fromkeys(stream[idx]):
        select=stream[idx]==name;person=name.split('|')[0].split('/')[-1]
        per.setdefault(person,[]).append((pc[select],labels[select]))
    archive=ROOT/'ni26_js321_20261117_active5/base_v66.zip'
    if sha(archive)!='9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed':raise ValueError('frozen package changed')
    with zipfile.ZipFile(archive) as z, np.load(io.BytesIO(z.read('weights.npz')),allow_pickle=False) as w:
        names=[str(v) for v in w['route_names']];base={k:w[k].copy(order='K') for k in w.files}
    if set(names)!=set(per):raise ValueError('source gallery name/order differs')
    helpers=owner_functions();heads=helpers['ps1_heads'](per,names,C=.1)
    prior=np.stack([np.mean([np.stack([helpers['stdz'](a[:,:128])[b==c].mean(0) for c in range(3)]) for a,b in per[q]],0) for q in names])
    path=MAIN/'studies/12nov/ownspace.py';spec=importlib.util.spec_from_file_location('source_fold_ownspace',path)
    own=importlib.util.module_from_spec(spec);spec.loader.exec_module(own)
    own.y=own.y.astype(np.int64).copy();own.y[held]=-999
    parts=[];source_rows={}
    for q in names:
        rows=train[sub[train]=='Dreyer2026Proteus/'+q]
        sessions=[rows[session[rows]==u] for u in dict.fromkeys(session[rows])]
        if not len(rows) or np.intersect1d(rows,query).size:raise ValueError('personal fit source violation')
        model=own.Personal(sessions)
        for clf in (model.T.steps[-1][1],model.C,model.S):
            if not np.array_equal(clf.classes_,np.arange(3)):raise ValueError('held label poison reached fit')
        parts.append(helpers['arrays'](model,own));source_rows[q]=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()
    context=dict(route_ps1_W=np.stack([heads[q].coef_ for q in names]),route_ps1_c=np.stack([heads[q].intercept_ for q in names]),
                 v6_tpri=prior,route_names=np.array(names))
    for k in parts[0]:
        if k in ('own_bands','own_beta','own_kref'):
            if not all(np.array_equal(v[k],parts[0][k]) for v in parts):raise ValueError('own-stage scalar mismatch')
            context[k]=parts[0][k]
        else:context[k]=np.concatenate([v[k] for v in parts])
    if np.array_equal(context['route_ps1_W'],base['route_ps1_W']) or np.array_equal(context['v6_tpri'],base['v6_tpri']):raise ValueError('fold injection inactive')
    meta=dict(status='fold_personal_context_source_fit_complete_ENGINEERING_ONLY',fold=fold,train_count=len(train),held_count=len(query),
              train_rows_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),held_rows_sha256=hashlib.sha256(query.astype('<i8').tobytes()).hexdigest(),
              person_source_hashes=source_rows,feature_basis_sha256=sha(file),source_sha256=sha('source.sha256'),seconds=time.monotonic()-start,
              query_labels_used=False,accuracy_computed=False,view_basis_scope='NEW original source-fit PC256 basis; native raw canary passed, complete raw8order bank separate',
              population_scope='frozen current-system scalar/event/calibration constants retained; not all-components-excluded validation',
              frozen_zip_sha256=sha(archive),names=names)
    out=Path('context_'+fold+'.npz');np.savez(out,**context,metadata=json.dumps(meta));meta['arrays_sha256']=sha(out)
    Path('context_'+fold+'.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':main(sys.argv[1])
