"""Only complete immutable person/seed matrix may open labels and call originalmetric."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import source332 as src

ARMS=tuple(n+v for n in 'PS' for v in ('F','M','V','B','D'))


def main():
    root=Path(os.environ['MOMENT_ROOT']);src.check()
    expected={f'{p}_s{s}.npz' for p in src.PEOPLE for s in (0,1)}
    if {p.name for p in (root/'output').glob('*.npz')}!=expected:raise ValueError('incomplete20foldmatrix')
    with np.load(src.data323.DATA) as z:subjects=z['subject']
    n=len(subjects);merged={(a,s,o):np.full((n,3),np.nan) for a in ARMS for s in (0,1) for o in src.ORDERS};hashes={};metadata={}
    for person in src.PEOPLE:
        rows=np.flatnonzero(np.char.endswith(subjects,'/'+person))
        for seed in (0,1):
            original=src.verify(person,seed);path=root/'output'/f'{person}_s{seed}.npz'
            with np.load(path) as z:
                m=json.loads(str(z['metadata']))
                if (m['held'],m['seed'],m['held_labels_opened'],m['source_sha256'])!=(person,seed,False,src.sha(root/'source.sha256')):raise ValueError('foldmetadata')
                if m['source']['original_fold']!=original or not np.array_equal(z['index'],rows):raise ValueError('foldsource/indexchanged')
                if set(z.files)!={'index','metadata'}|{f'{a}|{o}' for a in ARMS for o in src.ORDERS}:raise ValueError('incompletearm/order')
                for a in ARMS:
                    for o in src.ORDERS:
                        v=z[f'{a}|{o}']
                        if v.shape!=(len(rows),3) or not np.isfinite(v).all() or not np.allclose(v.sum(1),1,atol=1e-6):raise ValueError('invalidviewprobabilities')
                        merged[a,seed,o][rows]=v
            hashes[path.name]=src.sha(path);metadata[path.stem]=m
    if any(not np.isfinite(v).all() for v in merged.values()):raise ValueError('missingcanonicalrows')
    whole=src.PROD/'whole_results.json'
    if src.sha(whole)!='89dc9aa77860b8ba4f06925294565d50eb7cf5874f310d598bdad24b4d04a92d':raise ValueError('originalwholereportchanged')
    # All original80checkpoints/new20folds/everyview/order validatedbeforeheldlabels.
    import protocol,score_s_v66p as score
    if Path(protocol.__file__).resolve()!=(src.PROD/'experiments/protocol.py').resolve() or Path(score.__file__).resolve()!=(src.PROD/'studies/14nov/newmember/score_s_v66p.py').resolve():raise ValueError('wrongoriginalmetricmodule')
    d=protocol.load();y=d['y'];people=np.array([str(s).split('/')[-1] for s in d['subject']]);cell=np.char.add(np.char.add(d['stream'],'|'),d['interface'])
    counts=np.array([len(np.unique(cell[people==p])) for p in src.PEOPLE]);per={};absolute={}
    for a in ARMS:
        for o in src.ORDERS:per[a,o]=np.mean([score.person_accuracy(merged[a,k,o],y,cell,people,src.PEOPLE) for k in (0,1)],axis=0)
        values=np.stack([per[a,o] for o in src.ORDERS]);absolute[a]=dict(people_order=score.summarize(values,src.ORDERS,src.PEOPLE),official_equal_cell=score.cell_weighted_summary(values,src.ORDERS,counts))
    original=json.loads(whole.read_text())
    for label in 'PS':
        for view,old in (('F','F'),('B','C')):
            expected=original['official_equal_cell_scheme_equal'][f'{label}0{old}']
            if abs(absolute[label+view]['official_equal_cell']['mean']-expected)>1e-10:raise ValueError('originalendpointmetricnotreproduced')
    contrasts={};proxy=np.array([.43*.2 if p=='P04' else .43*.8 if p=='P10' else .57/8 for p in src.PEOPLE])
    for label in 'PS':
        forms={'M_minus_F':[('M',1),('F',-1)],'V_minus_F':[('V',1),('F',-1)],'B_minus_F':[('B',1),('F',-1)],'B_minus_D':[('B',1),('D',-1)],'interaction':[('B',1),('M',-1),('V',-1),('F',1)]}
        for name,terms in forms.items():
            delta=np.stack([sum(w*per[label+v,o] for v,w in terms) for o in src.ORDERS])
            contrasts[label+'_'+name]=dict(terms=terms,people_order=score.summarize(delta,src.ORDERS,src.PEOPLE),official_equal_cell=score.cell_weighted_summary(delta,src.ORDERS,counts),sensitivity57_43_not_returning_score=score.cell_weighted_summary(delta,src.ORDERS,proxy))
    result=dict(scope='selectedwholeinference-mechanism;notchain/returning/sealedgain',seed_aggregation='separate seedmetrics thenmean',
        source_sha256=src.sha(root/'source.sha256'),producer_source=src.SOURCE,absolute=absolute,contrasts=contrasts,
        hashes=hashes,fold_metadata=metadata,metric_sha256=src.sha(protocol.__file__),scorer_sha256=src.sha(score.__file__))
    with (root/'results.json').open('x') as out:json.dump(result,out,indent=2)
    print('OK:all20folds/80originalcheckpoints/80outputsperfold;originalendpointsreproduced;whole332metriccomplete',flush=True)


if __name__=='__main__':main()
