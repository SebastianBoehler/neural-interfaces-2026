"""Official equal-cell contrasts and paired nested people/order intervals (§381)."""
import json
from pathlib import Path
import numpy as np
from evaluate import TRAIN,TRAIN_SHA,owners,joint,source


def summarize(delta,counts):
    rng=np.random.default_rng(381)
    draws=rng.integers(4,size=(4000,4))
    schemes=[np.arange(4),np.arange(4,6),np.arange(6,8)]
    values=np.mean([delta[:,g].mean(1) for g in schemes],axis=0)
    boot=np.zeros_like(draws,dtype=float)
    for group in schemes:
        orders=rng.choice(group,size=(4000,4,len(group)))
        boot+=delta[draws[:,:,None],orders].mean(2)/3
    ci=np.percentile(np.average(boot,weights=counts[draws],axis=1),[10,90])
    return dict(official_pp=float(np.average(values,weights=counts)),paired80=ci.tolist(),
                by_person=dict(zip(joint.COHORT,values.tolist())),people_up=int((values>0).sum()))


def main():
    d,people,fit,runtime=source()
    fits=owners(fit)
    metric=joint.official()
    joint.pin(Path(metric.__file__),joint.METRIC_SHA)
    arms=['A6','population','frozen','JS']+[a+'_'+b for a in ('C','D','DC') for b in ('real','blank','donor')]
    scores={a:np.empty((4,8)) for a in arms}
    counts=np.empty(4,dtype=int)
    banks,proofs={},{}
    for i,p in enumerate(joint.COHORT):
        rows=np.flatnonzero(people==p)
        proof=json.loads(Path(p+'.json').read_text())
        assert proof['held']==p and proof['train_source']==TRAIN_SHA and proof['held_labels_opened'] is False
        assert tuple(proof['train_people'])==joint.FIT_PEOPLE and proof['train_rows']==joint.digest(fit)
        assert proof['actor_source']==joint.sha(Path('source.sha256')) and proof['test_rows']==joint.digest(rows)
        joint.pin(Path(p+'.npz'),proof['outputs_sha256'])
        with np.load(p+'.npz') as z:
            assert np.array_equal(z['index'],rows)
            assert set(z.files)=={'index'}|{a+'|'+o for a in arms for o in joint.ORDERS}
            banks[p]={k:z[k] for k in z.files if k!='index'}
        proofs[p]=proof
    before=dict(actors=proofs,source_only_training=fits,held_labels_opened=False,
                actor_source=joint.sha(Path('source.sha256')),metric_sha=joint.METRIC_SHA)
    Path('before_labels.json').write_text(json.dumps(before,indent=2))
    for i,p in enumerate(joint.COHORT):
        mask=people==p
        y=joint.labels(np.flatnonzero(mask),len(people))
        for j,o in enumerate(joint.ORDERS):
            for a in arms:
                v=banks[p][a+'|'+o]
                assert np.isfinite(v).all() and np.allclose(v.sum(1),1) and v.shape==(mask.sum(),3)
                result=metric.score(y,v.argmax(1),d,mask)
                scores[a][i,j]=100*result['balanced_accuracy_cell_mean']
                counts[i]=result['n_cells']
    assert np.all(counts==2)
    summary={a:summarize(s,counts)['official_pp'] for a,s in scores.items()}
    for a,v in [('A6',60.78519278107362),('population',53.82980510752687),('frozen',81.21561492579204),('JS',81.23552194696784)]:
        assert abs(summary[a]-v)<1e-9,(a,summary[a],v)
    pairs=[('A6','population'),('D_real','C_real'),('DC_real','D_real'),('DC_real','C_real')]
    for a in ('C','D','DC'):
        pairs.extend([(a+'_real',b) for b in ('population','A6','JS','frozen',a+'_blank',a+'_donor')])
    differences={a+'-'+b:summarize(scores[a]-scores[b],counts) for a,b in pairs}
    result=dict(summary=summary,contrasts=differences,eligible_people=joint.COHORT,
                source=joint.sha(Path('before_labels.json')))
    Path('result.json').write_text(json.dumps(result,indent=2))
    print(summary,flush=True)


if __name__=='__main__':
    main()
