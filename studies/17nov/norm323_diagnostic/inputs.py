"""Complete owning guards before labels; no replay, training or partial matrices."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

CHAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')
PRODUCER=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_normsplit_20261117')
PINS={'representation-source.sha256':'baf212bddfc79ed4d8bce9a9600c5c3db56d039d72c08f8fcfc92997dafbd55c',
      'results.txt':'9e82b8082948702b600687b455e9efcd027b1698ae21955deb9e5faa0196ae34',
      'slurm-3117040.out':'5faba4e1c72f60850a7e557fea817212b48db99497111cd263639d29bbed3bdf'}
SOURCE='88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5'
WHOLE='89dc9aa77860b8ba4f06925294565d50eb7cf5874f310d598bdad24b4d04a92d'
ARMS=('P0F','P0C','S0F','S0C')


def load():
    folder=CHAIN/'studies/14nov/newmember';sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('original323score',folder/'score_s_v66p.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    for name,pin in PINS.items():
        if s.sha256(CHAIN/name)!=pin:raise ValueError('chain/report identity changed:'+name)
    if (CHAIN/'slurm-3117040.out').read_text().splitlines()[-1]!='OK':raise ValueError('original replay not OK')
    for line in (CHAIN/'representation-source.sha256').read_text().splitlines():
        pin,name=line.split(maxsplit=1)
        if s.sha256(CHAIN/name)!=pin:raise ValueError('chain source changed')
    if s.sha256(PRODUCER/'source323.sha256')!=SOURCE or s.sha256(PRODUCER/'whole_results.json')!=WHOLE:
        raise ValueError('complete producer identity changed')
    for line in (PRODUCER/'source323.sha256').read_text().splitlines():
        pin,name=line.split(maxsplit=1)
        if s.sha256(PRODUCER/name)!=pin:raise ValueError('producer source changed')
    # Read only session metadata before original full provenance/zero/route/package collector.
    with np.load(CHAIN/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        sessions=len(np.unique(z['sess|all_loso']))
    outputs=s.collect(CHAIN,2655,sessions)
    seeds={};members={};pre={};clips={}
    validation=json.loads((PRODUCER/'runs/n323_streams/validation.json').read_text())
    for arm in ARMS:
        for order in s.ORDERS:
            values=[]
            for seed in (0,1):
                name=f'ttt18{order}_n323{arm.lower()}_base-all_loso-s{seed}.npz'
                path=PRODUCER/'runs/n323_streams'/name
                if s.sha256(path)!=validation['streams'][name]:raise ValueError('seedstream hash')
                with np.load(path,allow_pickle=False) as z:
                    m=json.loads(str(z['metadata']))
                    if (m['arm'],m['seed'],m['order'],m['source_manifest'])!=(arm,seed,order,SOURCE):raise ValueError('seed identity')
                    if m['folds']!=validation['folds'] or not np.array_equal(z['index'],np.arange(2655)):raise ValueError('fold/index')
                    value=z['proba'].copy()
                seeds[arm,seed,order]=value;values.append(value)
            expected=np.mean(values,axis=0)
            path=CHAIN/f'runs/representation_streams/ttt18{order}_n323{arm.lower()}_base-all_loso-ensemble2.npz'
            with np.load(path) as z:
                if not np.array_equal(z['proba'],expected):raise ValueError('stage is not exact2seedmean')
            clipped=np.clip(expected,1e-6,None);members[arm,order]=clipped/clipped.sum(1,keepdims=True)
            clips[f'{arm}|{order}']=int((expected<1e-6).sum())
    for arm in ('control',)+ARMS:
        for order in s.ORDERS:
            with np.load(CHAIN/f'runs/chainout/s281_v66p_{arm}_{order}.npz') as z:
                for base in s.BASES:
                    value=z[f'{base}|V6Fr'].copy()
                    if value.shape!=(2655,3) or not np.isfinite(value).all() or (value<0).any() or not np.allclose(value.sum(1),1,atol=1e-6):
                        raise ValueError('V6Fr is not probability data')
                    pre[arm,order,base]=value
    # Labels open only now, after complete matrices and original guards.
    with np.load(CHAIN/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        y=z['y|all_loso'];cell=z['cell|all_loso'];person=z['person|all_loso']
    return s,outputs,seeds,members,pre,y,cell,person,clips
