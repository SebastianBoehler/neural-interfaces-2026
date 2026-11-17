"""Saved original323weights; no producer label loading or newfit."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

PROD=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_normsplit_20261117')
SOURCE='88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5'
sys.path[:0]=[str(PROD),str(PROD/'experiments'),str(PROD/'studies/14nov/newmember')]
import model322,encoder312,modes,session_bn,data323,source_labels
from data323 import sha,ORDERS,PEOPLE,DATA_SHA
from verify_fold import verify
from encoder312 import fingerprint


def check():
    if sha(PROD/'source323.sha256')!=SOURCE:raise ValueError('original323sourcechanged')
    for line in (PROD/'source323.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        if sha(PROD/name)!=expected:raise ValueError('originalsourcefilechanged')
    for mod,name in ((model322,'model322.py'),(encoder312,'encoder312.py'),(modes,'modes.py'),(session_bn,'experiments/session_bn.py'),(data323,'data323.py'),(source_labels,'source_labels.py')):
        if Path(mod.__file__).resolve()!=(PROD/name).resolve():raise ValueError('wrong332dependencyimport')
    if sha(data323.DATA)!=DATA_SHA:raise ValueError('officialtensorchanged')


def load(held,seed,device):
    check();old=verify(held,seed)
    with np.load(data323.DATA,allow_pickle=False) as z:d={k:z[k] for k in ('X','subject','session','interface')}
    ids=np.array([str(s).split('/')[-1] for s in d['subject']]);test=np.flatnonzero(ids==held);fit=np.flatnonzero(ids!=held)
    d['stream']=np.char.add(np.char.add(d['subject'],'|'),d['session'])
    sessions=sorted(set(d['stream'][fit]),key=lambda s:(-np.sum(d['stream'][fit]==s),s))
    donor=np.flatnonzero(d['stream']==sessions[0])
    if not np.isin(donor,fit).all():raise ValueError('heldrowindonor')
    with np.load(PROD/'runs/shuffle_perms.npz') as z:perms={o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in ORDERS}
    result=PROD/'output'/f'{held}_s{seed}'/'result.npz'
    with np.load(result) as z:
        meta=json.loads(str(z['metadata']));original={f'{a}{m}|{o}':z[f'{a}0{m}|{o}'].copy() for a in 'PS' for m in 'FC' for o in ORDERS}
    nets={};checkpoints={}
    for label,norm in (('P','population'),('S','session')):
        path=result.parent/f'{label}0.pt'
        if sha(path)!=meta['checkpoints'][f'{label}0']:raise ValueError('checkpointchanged')
        payload=torch.load(path,map_location=device,weights_only=False)
        model322.SEED=seed;model=model322.Factor(norm,False,device)
        model.load_state_dict(payload['state'],strict=True);model.eval()
        if payload['norm']!=norm or payload['skip'] or payload['held']!=held or payload['seed']!=seed:raise ValueError('savedmodelidentity')
        nets[label]=(model,float(payload['scale']));checkpoints[label]=sha(path)
    return d,fit,test,donor,perms,nets,original,dict(original_fold=old,checkpoints=checkpoints,source=SOURCE,donor_session=sessions[0],donor_rows=donor.tolist())
