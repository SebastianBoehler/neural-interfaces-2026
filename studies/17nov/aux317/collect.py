"""Complete60fold gate, then existing official whole-model evaluator."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT=Path(os.environ['NI26_ROOT']); WS=Path.cwd()
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'studies/14nov/newmember')]
from data import ORDERS,PEOPLE,sha
from score_s_v66p import person_accuracy,summarize


def main():
    expected={f'{a}_s{s}_{p}.npz' for a in 'ABC' for s in (0,1) for p in PEOPLE}
    output=WS/'output'
    if {p.name for p in output.glob('*.npz')}!=expected:
        raise ValueError('incomplete60fold matrix')
    with np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False) as z:
        subjects=z['subject'].copy()
    n=len(subjects); lock=sha(WS/'source.sha256')
    data_sha=sha(ROOT/'data/proteus_windows.npz'); order_sha=sha(ROOT/'runs/shuffle_perms.npz')
    matrix={(a,s,o):np.full((n,3),np.nan) for a in 'ABC' for s in (0,1) for o in ORDERS}
    records={}; paired={}; inputs={}
    for name in sorted(expected):
        path=output/name
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']))
            arm,seed,held=m['arm'],m['seed'],m['held']
            if name!=f'{arm}_s{seed}_{held}.npz' or m['held_accuracy_computed']:
                raise ValueError('arm identity or outer score policy differs')
            rows=np.flatnonzero(np.char.endswith(subjects,'/'+held)); target=str(subjects[rows[0]])
            universe=sorted(set(subjects.tolist())); sources=[p for p in universe if p!=target]
            inner=[p for p in sources if p!=sources[seed]]
            if m['fit_people']!=sources or m['inner_fit_people']!=inner or m['validation_person']!=sources[seed]:
                raise ValueError('nested source split differs')
            if not np.array_equal(z['index'],rows) or not all(m['guard'].values()):
                raise ValueError('rows/numerical guards invalid')
            if (m['source_manifest'],m['data_sha256'],m['permutation_sha256'])!=(lock,data_sha,order_sha):
                raise ValueError('source/data/permutation provenance differs')
            if m['orders']!=list(ORDERS) or sha(path.with_suffix('.pt'))!=m['checkpoint_sha256']:
                raise ValueError('checkpoint/order provenance differs')
            external=m['shin']; external_path=Path(external['path'])
            if str(external_path) not in inputs:
                inputs[str(external_path)]=sha(external_path)
            if inputs[str(external_path)]!=external['sha256'] or len(external['recorded'])!=16:
                raise ValueError('external dataset/mask changed')
            if set(external['fit_people'])&set(external['validation_people']):
                raise ValueError('external source validation leakage')
            if arm in 'BC' and m['refit']['stopped_steps']==0 and arm=='B':
                raise ValueError('stopped auxiliary arm inactive')
            if arm=='B' and m['refit']['auxiliary_max_effective_gradient']!=0:
                raise ValueError('no-transfer arm leaked gradient')
            if arm=='C' and m['refit']['auxiliary_max_effective_gradient']<=0:
                raise ValueError('joint arm inactive')
            pair=paired.setdefault((held,seed),{})
            pair[arm]=(m['selected']['initial'],m['selected']['source_plan'],m['source_rows_sha256'],m['inner_rows_sha256'])
            for order in ORDERS:
                p=z['proba_'+order]
                if p.shape!=(len(rows),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):
                    raise ValueError('invalid complete posterior shard')
                matrix[arm,seed,order][rows]=p
        records[name]=sha(path)
    if any(len(set(v.values()))!=1 or set(v)!=set('ABC') for v in paired.values()):
        raise ValueError('matched main initial/exposure/row identities differ')
    if any(not np.isfinite(p).all() for p in matrix.values()):
        raise ValueError('incomplete canonical posterior matrix')
    # All complete/provenance checks above precede any outer-label evaluation.
    import protocol
    d=protocol.load(); y=d['y']; person=np.array([p.split('/')[-1] for p in subjects])
    cell=np.char.add(np.char.add(d['stream'],'|'),d['interface']); mask=np.ones(n,dtype=bool)
    official={}; per_person={}; seed_scores={}
    for arm in 'ABC':
        for order in ORDERS:
            values=[100*protocol.score(y,matrix[arm,s,order].argmax(1),d,mask)['balanced_accuracy_cell_mean'] for s in (0,1)]
            seed_scores[arm,order]=values; official[arm,order]=float(np.mean(values))
            per_person[arm,order]=np.mean([person_accuracy(matrix[arm,s,order],y,cell,person,PEOPLE) for s in (0,1)],0)
    report=dict(status='complete60fold_guarded',hashes=records,producer_source_manifest=lock,
                data_sha256=data_sha,permutation_sha256=order_sha,external_inputs=inputs,
                official_equal_cell={a:{o:official[a,o] for o in ORDERS} for a in 'ABC'},
                per_seed_official={a:{o:seed_scores[a,o] for o in ORDERS} for a in 'ABC'},
                effects={f'{a}_minus_{b}':summarize(np.stack([per_person[a,o]-per_person[b,o] for o in ORDERS]),ORDERS,PEOPLE)
                         for a,b in [('C','B'),('C','A'),('B','A')]},
                scope='whole new-person systems;seed-argmax metrics averaged;not two-seed ensemble/current-chain/returning/worker')
    with (WS/'whole_results.json').open('x') as f:
        json.dump(report,f,indent=2)
    print('complete60fold matrix validated and whole-model scored',flush=True)


if __name__=='__main__':
    main()
