"""Assemble only all ten locked target folds into core-member deltas."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(os.environ['NI26_WS']).resolve(strict=True)
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
NAMES={'Anchor':'anchor','Fixed':'fixed','Scalar':'scalar','Blocks':'blocks',
       'Scalar0':'scalar0','Scalar1':'scalar1','Blocks0':'blocks0','Blocks1':'blocks1'}
ARCHIVES={o:('coremember.npz' if o in ('within0','within1','full0','runs0') else 'coremember_within2_within3_full1_runs1.npz') for o in ORDERS}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    people=[f'P{i:02d}' for i in range(1,11)]
    with np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False) as d:
        subject=np.asarray([str(v).rsplit('/',1)[-1] for v in d['subject']]); n=len(subject)
    with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z:
        perms={o:z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    arrays={(a,o):np.full((n,3),np.nan) for a in ('Rzero',*NAMES) for o in ORDERS}
    shards={}; source_sha=sha(ROOT/'target-src.sha256')
    meta_hashes={name:digest for digest,name in (line.split() for line in (ROOT/'meta.sha256').read_text().splitlines())}
    for held in people:
        path=ROOT/f'runs/s307/target/{held}.npz'; before=sha(path)
        if path.with_suffix('.cpu-route').read_text().split()!=['CPU',before,source_sha]: raise ValueError(f'{held}: route mismatch')
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata'].item())); rows=z['rows']
            if m['held']!=held or m['held_accuracy_computed'] is not False or not np.array_equal(rows,np.flatnonzero(subject==held)) or m['fit_gradmax']>1e-5 or m['max_delta_ratio']>.03000001:
                raise ValueError(f'{held}: invalid fold')
            for kind in ('scalar','blocks'):
                for seed in (0,1):
                    name=f'runs/s307/meta/{held}_{kind}_s{seed}.npz'
                    if m['models_sha256'][f'{kind}_s{seed}']!=meta_hashes[name]: raise ValueError(f'{held}: changed model')
            for order in ORDERS:
                if not np.array_equal(np.sort(perms[order]),np.arange(n)) or m['updates'][order]<=0: raise ValueError('order/update guard')
                with np.load(ROOT/'runs'/ARCHIVES[order],allow_pickle=False) as ar: reference=ar[f'{order}|all_loso'][rows]
                for arm,key in NAMES.items():
                    p=z[f'{key}|{order}']; delta=p-reference
                    if p.shape!=(len(rows),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0):
                        raise ValueError(f'{held}/{arm}/{order}: invalid posterior')
                    arrays[arm,order][rows]=delta
                arrays['Rzero',order][rows]=0.
        if sha(path)!=before: raise ValueError('target shard changed during read')
        shards[path.name]=before
    if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError('incomplete target matrix')
    for order in ORDERS:
        for arm in ('Fixed','Scalar','Blocks','Scalar0','Scalar1','Blocks0','Blocks1'):
            if np.array_equal(arrays[arm,order],arrays['Anchor',order]):
                raise ValueError(f'{arm}/{order}: update changed no member posterior')
    out=ROOT/'runs/s301_native/deltas'; out.mkdir(parents=True,exist_ok=False); files={}
    for (arm,order),delta in arrays.items():
        name=f'{arm}_{order}.npz'; path=out/name
        meta={'arm':arm,'order':order,'order_sha256':hashlib.sha256(perms[order].tobytes()).hexdigest(),'producer_source_sha256':source_sha}
        with path.open('xb') as f: np.savez_compressed(f,delta=delta,metadata=json.dumps(meta,sort_keys=True))
        files[name]=sha(path)
    manifest=out.parent/'manifest.json'
    with manifest.open('x') as f: json.dump({'files':files,'shards':shards,'producer_source_sha256':source_sha,'meta_manifest_sha256':sha(ROOT/'meta.sha256')},f,sort_keys=True,indent=2)
    print('complete 72-delta target matrix pinned; no target score',flush=True)


if __name__=='__main__':main()
