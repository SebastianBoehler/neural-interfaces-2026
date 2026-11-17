"""Source-only task maps on unchanged original external teacher heads/production input prior."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from returning_bank import ROOT,input_class,sha


def digest(a):return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()


def main():
    torch.set_num_threads(1);start=time.monotonic();package=ROOT/'ni26_js321_20261117_orchestration4/prepared_jsall'
    original=package/'teacher.pt'
    if sha(original)!='98f7df456dcbecaaadb2e0c0a09a130dc6fd3e2de187adf606dd1ce1b4330b5c':raise ValueError('frozen external teacher changed')
    model=torch.jit.load(str(original),map_location='cpu').eval();heads_before={k:v.clone() for k,v in model.heads.state_dict().items()}
    with np.load(package/'weights.npz') as z:p={k:z['net34_'+k].copy() for k in ('align','k','G','scale','shrink')}
    data=ROOT/'neural-interfaces-2026/data/proteus_windows.npz'
    if sha(data)!='2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327':raise ValueError('data changed')
    with np.load(data,allow_pickle=True) as z:X=z['X'].copy();sub=z['subject'].copy();session=z['session'].astype(str);y=z['y'].copy()
    Input=input_class();stream=sub+'|'+session;stim=np.flatnonzero((sub=='Dreyer2026Proteus/P04')&(session=='1'))[:12]
    ins=Input(p);h=np.stack([ins(X[r]) for r in stim]);t=time.monotonic()
    with torch.no_grad():
        for raw in h:model.heads(torch.from_numpy(raw[None]))
    b1_seconds=time.monotonic()-t
    if b1_seconds>5.:raise ValueError(f'prospective CPU12window5s cost stop {b1_seconds}')
    features=np.full((len(X),38),np.nan);t=time.monotonic()
    with torch.no_grad():
        for s in dict.fromkeys(stream):
            ins=Input(p)
            for r in np.flatnonzero(stream==s):features[r]=model.heads(torch.from_numpy(ins(X[r])[None]))[0].double().numpy()
    feature_seconds=time.monotonic()-t
    if not np.isfinite(features).all():raise ValueError('nonfinite frozen external features')
    records=[]
    for person in ('P04','P10'):
        fold=person+'_s23';held=(sub=='Dreyer2026Proteus/'+person)&(session!='1');train=np.flatnonzero(~held);query=np.flatnonzero(held)
        labels=y.copy();labels[held]=-999
        if len(train)!=(2278 if person=='P04' else 2270) or np.intersect1d(train,query).size:raise ValueError('taskmap source mask differs')
        lr=LogisticRegression(C=.01,max_iter=3000,class_weight='balanced').fit(features[train],labels[train])
        if not np.array_equal(lr.classes_,np.arange(3)):raise ValueError('poison reached taskmap/source classes differ')
        with torch.no_grad():model.lin.weight.copy_(torch.from_numpy(lr.coef_).float());model.lin.bias.copy_(torch.from_numpy(lr.intercept_).float())
        if not all(torch.equal(v,model.heads.state_dict()[k]) for k,v in heads_before.items()):raise ValueError('external heads changed')
        ref=lr.predict_proba(features[stim]);ins=Input(p)
        with torch.no_grad():actual=torch.cat([model(torch.from_numpy(ins(X[r])[None])) for r in stim]).double().softmax(1).numpy()
        error=float(np.abs(ref-actual).max())
        if error>1e-4:raise ValueError('new source teacher taskmap eager/trace export failed')
        out=Path(fold);out.mkdir();model.save(str(out/'teacher.pt'));loaded=torch.jit.load(str(out/'teacher.pt')).eval();ins=Input(p)
        with torch.no_grad():repeat=torch.cat([loaded(torch.from_numpy(ins(X[r])[None])) for r in stim]).double().softmax(1).numpy()
        if not np.array_equal(repeat,actual):raise ValueError('saved taskmap samepath differs')
        proof=dict(status='new_source_teacher_taskmap_export_passed_NOT_accuracy',fold=fold,train_count=len(train),held_count=len(query),
                   train_rows_sha256=digest(train),held_rows_sha256=digest(query),external_teacher_sha256=sha(original),taskmap_C=.01,class_weight='balanced',
                   teacher_sha256=sha(out/'teacher.pt'),source_sha256=sha('source.sha256'),export_maxabs=error,saved_samepath_bitwise=True,
                   external_heads_immutable=True,held_labels_poisoned=True,accuracy_computed=False,
                   scope='NEW source-only task map;frozen external heads and original global production input G/scale retained explicitly conditional')
        (out/'proof.json').write_text(json.dumps(proof,indent=2));records.append(proof)
    Path('teacher_complete.json').write_text(json.dumps(dict(status='both_source_teacher_maps_complete_NOT_accuracy',folds=records,
        b1_12source_seconds=b1_seconds,feature_seconds=feature_seconds,seconds=time.monotonic()-start,accuracy_computed=False),indent=2))


if __name__=='__main__':main()
