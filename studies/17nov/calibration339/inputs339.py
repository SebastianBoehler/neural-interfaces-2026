"""Exactly323 trainer,332 views; selected source labels, never outer labels."""
import copy
import sys
from pathlib import Path
import numpy as np
import torch
MOMENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_moments332_v2_20261117')
sys.path.insert(0,str(MOMENT))
import source332 as src
import views332
from views332 import prepare
from encoder312 import fingerprint
SHA332='6c0f967ca5e8a7240b676de483bdc575eab2918c526fee4b832dc107d925abab'
OUTER='P01';CAL='P02';VIEWS=('F','B','V')


def inputs():
    src.check()
    if Path(views332.__file__).resolve()!=(MOMENT/'views332.py').resolve():raise ValueError('wrong332viewsimport')
    if src.sha(MOMENT/'source.sha256')!=SHA332:raise ValueError('332sourceidentity')
    for line in (MOMENT/'source.sha256').read_text().splitlines():
        h,n=line.split(maxsplit=1)
        if src.sha(MOMENT/n)!=h:raise ValueError('332sourcechanged')
    with np.load(src.data323.DATA,allow_pickle=False) as z:d={k:z[k] for k in ('X','subject','session','interface')}
    people=np.array([str(x).split('/')[-1] for x in d['subject']]);fit=np.flatnonzero(~np.isin(people,[OUTER,CAL]));cal=np.flatnonzero(people==CAL)
    d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session']);d['y']=np.full(len(people),-1,np.int64)
    d['y'][fit]=src.source_labels.read_training_labels(src.data323.DATA,fit,len(people))
    if set(people[fit])!={f'P{i:02d}' for i in range(3,11)} or np.any(d['y'][people==OUTER]!=-1) or np.any(d['y'][cal]!=-1):raise ValueError('nestedsourceexclusion')
    with np.load(src.PROD/'runs/shuffle_perms.npz') as z:perms={o:z[f'{o[:-1]}|{o[-1]}|all_loso'].copy() for o in src.ORDERS}
    for p in perms.values():
        if not np.array_equal(np.sort(p),np.arange(len(people))):raise ValueError('pinnedorderinvalid')
    return d,fit,cal,people,perms


def logits(model,scale,d,rows,perms,device):
    probe=torch.as_tensor(d['X'][rows[:10]]/scale,device=device).float();views,_=prepare(model,probe)
    out={};checks={};position=np.full(len(d['X']),-1,int);position[rows]=np.arange(len(rows))
    for name in VIEWS:
        view=views[name];before=fingerprint(view);err=0.
        for order,p in perms.items():
            rank=np.empty(len(p),int);rank[p]=np.arange(len(p));result=np.full((len(rows),3),np.nan)
            for session in dict.fromkeys(d['stream'][rows]):
                r=rows[d['stream'][rows]==session];r=r[np.argsort(rank[r],kind='stable')]
                def call(indices):
                    with torch.no_grad():return view(torch.as_tensor(d['X'][indices]/scale,device=device).float()).double().cpu().numpy()
                q=call(r);prefix=call(r[:10]);err=max(err,float(abs(q[:len(prefix)]-prefix).max()))
                if not np.allclose(q[:len(prefix)],prefix,atol=2e-5,rtol=2e-5) or not np.allclose(q[:1],call(r[:1]),atol=2e-5,rtol=2e-5):raise ValueError('logitprefix/B1')
                other=rows[d['stream'][rows]!=session]
                if len(other):call(other[:10])
                if not np.array_equal(prefix,call(r[:10])):raise ValueError('sessionreset')
                with torch.no_grad():copied=copy.deepcopy(view)(torch.as_tensor(d['X'][r[:10]]/scale,device=device).float()).double().cpu().numpy()
                if not np.array_equal(prefix,copied):raise ValueError('deepcopylogitguard')
                result[position[r]]=q
            if not np.isfinite(result).all():raise ValueError('incompletecalibrationbank')
            out[name+'|'+order]=result
        if fingerprint(view)!=before:raise ValueError('viewmutatedweights/buffers')
        checks[name]=dict(prefix_max_abs=err,state_sha256=before)
    if not any(np.max(abs(out['F|'+o]-out['V|'+o]))>1e-7 for o in perms):raise ValueError('varianceviewinactive')
    return out,checks
