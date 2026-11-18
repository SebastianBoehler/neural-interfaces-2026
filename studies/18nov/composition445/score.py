"""Compare source-only joint fits on four paired people and eight historical delivery orders."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from actor import ARMS, ARCH, ROWS, SEEDS, WORK, joint, prior

spec=importlib.util.spec_from_file_location('summary381',WORK/'ni26_onebn381_actor_v1_20261118/score.py')
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def main():
    d,people,_,_=prior.source()
    records={}
    for person in joint.COHORT:
        r=json.loads(Path(person+'.json').read_text())
        assert r['held']==person and r['train']==ROWS and r['test']==joint.digest(np.flatnonzero(people==person))
        for f,k in [(person+'.npz','output'),('fits.json','fits'),('source.sha256','source')]:
            joint.pin(Path(f),r[k])
        records[person]=r
    Path('before_labels.json').write_text(json.dumps(records,indent=2))
    metric=joint.official()
    joint.pin(Path(metric.__file__),joint.METRIC_SHA)
    counts=np.full(4,2)
    results={}
    for seed in SEEDS:
        vals={a:np.empty((4,8)) for a in ARMS}
        for i,p in enumerate(joint.COHORT):
            mask=people==p
            rows=np.flatnonzero(mask)
            gold=joint.labels(rows,len(people))
            with np.load(p+'.npz') as z:
                assert np.array_equal(z['index'],rows)
                for j,o in enumerate(joint.ORDERS):
                    for a in ARMS:
                        pred=z[f'{seed}|{a}|{o}']
                        assert np.isin(pred,[0,1,2]).all()
                        v=metric.score(gold,pred,d,mask)
                        assert v['n_cells']==2
                        vals[a][i,j]=100*v['balanced_accuracy_cell_mean']
        means={a:summary.summarize(v,counts)['official_pp'] for a,v in vals.items()}
        hist=json.loads((ARCH[seed][0]/'result.json').read_text())['summary']
        assert all(abs(means[a]-hist[a])<1e-9 for a in ('A6','J_real','frozen'))
        pairs = [('skew_aux_real',a) for a in ('skew_ce_real','random_aux_real','A6','J_real',
                                               'skew_aux_blank','skew_aux_donor')]
        pairs.append(('random_aux_real','random_ce_real'))
        gains = {a+'-'+b:summary.summarize(vals[a]-vals[b],counts) for a,b in pairs}
        gains['skew_specific_interaction'] = summary.summarize(
            vals['skew_aux_real']-vals['skew_ce_real']-vals['random_aux_real']+vals['random_ce_real'],counts)
        results[seed]=dict(means=means,contrasts=gains)
    Path('result.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
