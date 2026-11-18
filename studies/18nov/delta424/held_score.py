"""Compare the whole learned decoder under paired official cell/person/order scoring."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from fits import ARCH,ARMS,REFERENCES,ROWS,SEEDS,WORK,joint,prior,sha

base=WORK/'ni26_onebn381_actor_v1_20261118'
sys.path.insert(0,str(base))
spec=importlib.util.spec_from_file_location('summaries381',base/'score.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    d,people,_,_=prior.source()
    arms=(*ARMS,*REFERENCES)
    vals={s:{a:np.empty((4,8)) for a in arms} for s in SEEDS}
    records={}
    predictions={}
    for p in joint.COHORT:
        mask=people==p
        r=json.loads(Path(p+'.json').read_text())
        assert r['held']==p and r['held_labels_opened'] is False and r['train_rows']==ROWS
        assert r['test_rows']==joint.digest(np.flatnonzero(mask)) and r['fits']==sha(Path('fits.json'))
        assert r['source']==sha(Path('source.sha256')) and r['outputs']==sha(Path(p+'.npz'))
        with np.load(p+'.npz') as z:
            assert np.array_equal(z['index'],np.flatnonzero(mask))
            assert set(z.files)=={'index'}|{f'{s}|{a}|{o}' for s in SEEDS for a in arms for o in joint.ORDERS}
            predictions[p]={k:z[k] for k in z.files if k!='index'}
        records[p]=r
    before=Path('before_labels.json')
    before.write_text(json.dumps(dict(actors=records,held_labels_opened=False),indent=2))
    metric=joint.official()
    joint.pin(Path(metric.__file__),joint.METRIC_SHA)
    counts=np.full(4,2)
    for i,p in enumerate(joint.COHORT):
        mask=people==p
        gold=joint.labels(np.flatnonzero(mask),len(people))
        for s in SEEDS:
            for a in arms:
                for j,o in enumerate(joint.ORDERS):
                    v=predictions[p][f'{s}|{a}|{o}']
                    assert np.isin(v,[0,1,2]).all()
                    r=metric.score(gold,v,d,mask)
                    assert r['n_cells']==counts[i]
                    vals[s][a][i,j]=100*r['balanced_accuracy_cell_mean']
    out={}
    for seed,v in vals.items():
        sums={a:module.summarize(m,counts)['official_pp'] for a,m in v.items()}
        historical=json.loads((ARCH[seed][0]/'result.json').read_text())['summary']
        assert all(abs(sums[k]-historical[k])<1e-9 for k in REFERENCES)
        pairs=[('T',a) for a in arms if a!='T']+[('S','C')]
        out[seed]=dict(summary=sums,contrasts={a+'-'+b:module.summarize(v[a]-v[b],counts) for a,b in pairs})
    Path('held_result.json').write_text(json.dumps(dict(seeds=out,before=sha(before)),indent=2))


if __name__=='__main__':
    main()
