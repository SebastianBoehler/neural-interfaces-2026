"""Compare preserved A6 behavior and held official accuracy against matched controls."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from actor import ARCH,ARMS,REFERENCES,ROWS,SEEDS,WORK,joint,prior,sha
base=WORK/'ni26_onebn381_actor_v1_20261118'
spec=importlib.util.spec_from_file_location('summaries381',base/'score.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    d,people,_,_=prior.source()
    arms=ARMS+REFERENCES
    vals={s:{a:np.empty((4,8)) for a in arms} for s in SEEDS}
    records,outputs={},{}
    for p in joint.COHORT:
        r=json.loads(Path(p+'.json').read_text())
        assert not r['held_labels_opened'] and r['held']==p and r['train_rows']==ROWS
        assert r['test_rows']==joint.digest(np.flatnonzero(people==p)) and r['fits']==sha('fits.json')
        assert r['source']==sha('source.sha256') and r['outputs']==sha(p+'.npz') and r['probs']==sha(p+'_prob.npz')
        with np.load(p+'.npz') as z:
            assert np.array_equal(z['index'],np.flatnonzero(people==p))
            assert set(z.files)=={'index'}|{f'{s}|{a}|{o}' for s in SEEDS for a in arms for o in joint.ORDERS}
            outputs[p]={k:z[k] for k in z.files if k!='index'}
        records[p]=r
    Path('before_labels.json').write_text(json.dumps(records,indent=2))
    metric=joint.official()
    joint.pin(Path(metric.__file__),joint.METRIC_SHA)
    counts=np.full(4,2)
    for i,p in enumerate(joint.COHORT):
        m=people==p
        y=joint.labels(np.flatnonzero(m),len(people))
        for s in SEEDS:
            for a in arms:
                for j,o in enumerate(joint.ORDERS):
                    score=metric.score(y,outputs[p][f'{s}|{a}|{o}'],d,m)
                    assert score['n_cells']==2
                    vals[s][a][i,j]=100*score['balanced_accuracy_cell_mean']
    out={}
    for s,v in vals.items():
        metrics={a:module.summarize(arr,counts)['official_pp'] for a,arr in v.items()}
        history=json.loads((ARCH[s][0]/'result.json').read_text())['summary']
        assert all(abs(metrics[k]-history[k])<1e-9 for k in REFERENCES)
        pairs=[(b+'T',c) for b in 'JK' for c in (b+'P',b+'U','A6','JS')]
        pairs += [('JU','J_real'),('KU','K'),('JT','KT')]
        out[s]=dict(metrics=metrics,contrasts={a+'-'+b:module.summarize(v[a]-v[b],counts) for a,b in pairs})
    Path('result.json').write_text(json.dumps(dict(seeds=out,proof=sha('before_labels.json'),fits=sha('fits.json')),indent=2))


if __name__=='__main__':
    main()
