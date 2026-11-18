"""Verify all held banks and reference controls, then score paired official cells."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from actor import ARCH, ARMS, PROTO_SHA, ROWS, WORK, joint, prior

spec=importlib.util.spec_from_file_location('summary381',WORK/'ni26_onebn381_actor_v1_20261118/score.py')
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def main():
    d,people,_,_=prior.source()
    records={}
    for p in joint.COHORT:
        r=json.loads(Path(p+'.json').read_text())
        test=np.flatnonzero(people==p)
        assert (r['held'],r['train'],r['test'])==(p,ROWS,joint.digest(test))
        assert r['prototypes']==PROTO_SHA and r['held_labels_opened'] is False
        joint.pin(Path(p+'.npz'),r['output'])
        joint.pin(Path('actor.sha256'),r['source'])
        with np.load(p+'.npz') as z:
            assert np.array_equal(z['index'],test)
            assert set(z.files)=={'index'}|{f'{s}|{a}|{o}' for s in ARCH for a in ARMS for o in joint.ORDERS}
        records[p]=r
    Path('before_labels.json').write_text(json.dumps(records,indent=2))
    metric=joint.official()
    joint.pin(Path(metric.__file__),joint.METRIC_SHA)
    result={}
    for seed,(root,_) in ARCH.items():
        vals={a:np.empty((4,8)) for a in ARMS}
        changes={k:0 for k in ('semantic-unchanged','wrong-unchanged','semantic-wrong')}
        for i,p in enumerate(joint.COHORT):
            mask=people==p
            y=joint.labels(np.flatnonzero(mask),len(people))
            with np.load(p+'.npz') as z:
                for j,o in enumerate(joint.ORDERS):
                    for a in ARMS:
                        pred=z[f'{seed}|{a}|{o}']
                        assert np.isin(pred,[0,1,2]).all()
                        v=metric.score(y,pred,d,mask)
                        assert v['n_cells']==2
                        vals[a][i,j]=100*v['balanced_accuracy_cell_mean']
                    for key in changes:
                        left,right=key.split('-')
                        changes[key]+=int((z[f'{seed}|{left}_real|{o}']!=z[f'{seed}|{right}_real|{o}']).sum())
        counts=np.full(4,2)
        means={a:summary.summarize(v,counts)['official_pp'] for a,v in vals.items()}
        history=json.loads((root/'result.json').read_text())['summary']
        for a in ('frozen','population','A6','J_real','J_blank','J_donor'):
            assert abs(means[a.replace('J_','unchanged_')]-history[a]) < 1e-9
        pairs=[('semantic_real',a) for a in ('unchanged_real','wrong_real','semantic_donor','semantic_blank')]
        pairs.extend([('wrong_real','unchanged_real'),('semantic_donor','unchanged_donor')])
        contrasts={a+'-'+b:summary.summarize(vals[a]-vals[b],counts) for a,b in pairs}
        result[seed]=dict(means=means,contrasts=contrasts,changed_decisions=changes)
    Path('result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
