"""Score registered whole-model contrasts on four held people and eight deliveries."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from actor382 import ARMS, PRIOR, joint, prior, sha, verify

sys.path.insert(0,str(PRIOR))
spec=importlib.util.spec_from_file_location('score381',PRIOR/'score.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def main(root,seed,expected):
    verify(root,seed,expected)
    d,people,rows,_=prior.source()
    metric=joint.official()
    joint.pin(Path(metric.__file__),joint.METRIC_SHA)
    scores={a:np.empty((4,8)) for a in ARMS}
    counts=np.empty(4,dtype=int)
    banks={}
    proofs={}
    for p in joint.COHORT:
        test=np.flatnonzero(people==p)
        r=json.loads(Path(p+'.json').read_text())
        assert r['seed']==seed and r['fit_lock']==expected and r['held']==p
        assert tuple(r['train_people'])==joint.FIT_PEOPLE and r['train_rows']==joint.digest(rows)
        assert r['actor_source']==sha(Path('source.sha256')) and r['test_rows']==joint.digest(test)
        assert r['held_labels_opened'] is False
        joint.pin(Path(p+'.npz'),r['outputs_sha256'])
        with np.load(p+'.npz') as z:
            assert np.array_equal(z['index'],test)
            assert set(z.files)=={'index'}|{a+'|'+o for a in ARMS for o in joint.ORDERS}
            banks[p]={k:z[k] for k in z.files if k!='index'}
        proofs[p]=r
    before=dict(actors=proofs,fit_lock=expected,held_labels_opened=False,metric_sha=joint.METRIC_SHA)
    Path('before_labels.json').write_text(json.dumps(before,indent=2))
    for i,p in enumerate(joint.COHORT):
        mask=people==p
        y=joint.labels(np.flatnonzero(mask),len(people))
        for j,o in enumerate(joint.ORDERS):
            for a in ARMS:
                v=banks[p][a+'|'+o]
                assert np.isfinite(v).all() and np.allclose(v.sum(1),1) and v.shape==(mask.sum(),3)
                r=metric.score(y,v.argmax(1),d,mask)
                scores[a][i,j]=100*r['balanced_accuracy_cell_mean']
                counts[i]=r['n_cells']
                assert counts[i]==2
    summary={a:old.summarize(v,counts)['official_pp'] for a,v in scores.items()}
    historical=json.loads((PRIOR/'result.json').read_text())['summary']
    for a in ('population','A6','frozen','JS','C_real','D_real','DC_real'):
        assert abs(summary[a]-historical[a])<1e-9
    pairs=[('J_real',b) for b in ARMS if b!='J_real']
    pairs += [('K','A6'),('K','JS'),('F_real','D_real'),('F_real','F_blank'),('F_real','F_donor')]
    diff={a+'-'+b:old.summarize(scores[a]-scores[b],counts) for a,b in pairs}
    out=dict(seed=seed,summary=summary,contrasts=diff,source=sha(Path('before_labels.json')))
    Path('result.json').write_text(json.dumps(out,indent=2))
    print(summary,flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
