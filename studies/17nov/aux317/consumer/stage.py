"""Strict complete317producer adapter for the unchanged §315 consumer machinery."""
import json
import os
from pathlib import Path

import numpy as np
from config import GROUPS,ORDERS,sha


def main():
    root=Path(os.environ['NI26_WS']).resolve(strict=True)
    config=json.loads((root/'representation_config.json').read_text())
    if config['group']!='317':
        raise ValueError('317consumer config required')
    source=Path(config['producer_path']).resolve(strict=True)
    lock=source/'source.sha256'
    if sha(lock)!=config['producer_source']:
        raise ValueError('producer source lock changed')
    summary=source/'whole_results.json'; report=json.loads(summary.read_text())
    expected={f'{a}_s{s}_P{p:02d}.npz' for a in 'ABC' for s in (0,1) for p in range(1,11)}
    if report['status']!='complete60fold_guarded' or set(report['hashes'])!=expected:
        raise ValueError('successful complete60fold collector required')
    if report['producer_source_manifest']!=config['producer_source']:
        raise ValueError('collector producer source differs')
    inputs={str(lock):sha(lock),str(summary):sha(summary)}
    with np.load(root/'data/proteus_windows.npz',allow_pickle=False) as z:
        subjects=z['subject'].astype(str)
    n=len(subjects); matrix={(a,s,o):np.full((n,3),np.nan) for a in 'ABC' for s in (0,1) for o in ORDERS}
    for name,digest in report['hashes'].items():
        path=source/'output'/name
        if sha(path)!=digest:
            raise ValueError('complete producer shard changed')
        inputs[str(path)]=digest
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata'])); a,s,held=m['arm'],m['seed'],m['held']
            rows=np.flatnonzero(np.char.endswith(subjects,'/'+held)); target=subjects[rows[0]]
            if name!=f'{a}_s{s}_{held}.npz' or m['smoke'] or m['held_accuracy_computed']:
                raise ValueError('producer identity/scope differs')
            if not np.array_equal(z['index'],rows) or target in m['fit_people'] or target in m['inner_fit_people']:
                raise ValueError('producer source/canonical rows differ')
            if len(m['fit_people'])!=9 or len(m['inner_fit_people'])!=8 or m['source_manifest']!=config['producer_source']:
                raise ValueError('source exclusion/manifest differs')
            if sha(path.with_suffix('.pt'))!=m['checkpoint_sha256']:
                raise ValueError('producer checkpoint changed')
            inputs[str(path.with_suffix('.pt'))]=m['checkpoint_sha256']
            for o in ORDERS:
                p=z['proba_'+o]
                if p.shape!=(len(rows),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1):
                    raise ValueError('invalid probability shard')
                matrix[a,s,o][rows]=p
    for path,digest in report['external_inputs'].items():
        if sha(path)!=digest:
            raise ValueError('external training input changed')
        inputs[path]=digest
    directory=root/'runs/representation_streams'; directory.mkdir(exist_ok=False)
    target=root/'runs/representation_inputs'; target.mkdir(exist_ok=False); records={}
    for a,family in zip('ABC',GROUPS['317']['families']):
        for o in ORDERS:
            p=np.mean([matrix[a,s,o] for s in (0,1)],axis=0)
            if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):
                raise ValueError('invalid complete two-seed ensemble')
            metadata=dict(family=family,order=o,seeds=[0,1],operation='arithmetic_probability_mean',
                          use='same fixed ensemble in all six incumbent CP bases',producer_source=config['producer_source'])
            path=directory/f'ttt18{o}_{family}_base-all_loso-ensemble2.npz'
            with path.open('xb') as f:
                np.savez(f,index=np.arange(n),proba=p,metadata=json.dumps(metadata))
            records[path.name]=sha(path)
    manifest=target/'streams.sha256'
    manifest.write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(records.items())))
    (target/'lock.json').write_text(json.dumps(dict(group='317',manifest_sha256=sha(manifest),producer_inputs_sha256=inputs),indent=2))
    print('complete317matrix staged;24 two-seed ensemble streams;no scores read',flush=True)


if __name__=='__main__':
    main()
