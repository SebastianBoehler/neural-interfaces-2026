"""Stratified paired participant and delivery-order contrasts for external Wang data."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from wang_inputs import orders

PEOPLE = [8,9,22,23]
ARMS = ['old','cap','A6','pop']
POSITIONS = np.array([[5,4,3],[5,4,3],[3,5,4],[3,5,4]])
rng = np.random.default_rng(434)
person_boot = np.concatenate([rng.integers(i,i+2,(10000,2)) for i in (0,2)],axis=1)
order_boot = rng.integers(0,2,(10000,4,2))


def stat(x):
    assert x.shape == (4,2)
    sampled = x[person_boot[:,:,None],order_boot]
    def measure(z, b):
        return dict(mean=z.mean(),ci80=np.quantile(b.mean((1,2)),[.1,.9]).tolist())
    sequences = [measure(x[i:i+2],sampled[:,i:i+2]) for i in (0,2)]
    gap = sampled[:,2:].mean((1,2))-sampled[:,:2].mean((1,2))
    return dict(**measure(x,sampled), per_person=x.mean(1).tolist(),sequence=sequences,
                sequence_difference=dict(mean=x[2:].mean()-x[:2].mean(),ci80=np.quantile(gap,[.1,.9]).tolist()))



def additive(contrast):
    # Each person has a fixed intercept. Visits 1 and first cognitive position are baselines.
    visits = np.tile([1,2,3],4)
    positions = POSITIONS.ravel()
    design = np.column_stack([np.repeat(np.eye(4),3,axis=0), visits==2,visits==3,
                              positions==4,positions==5])
    assert np.linalg.matrix_rank(design) == 8
    fit = np.linalg.pinv(design)
    values = contrast.mean(1).ravel()
    # Stratified resampling keeps both schedules in exactly the original design slots.
    boot = contrast[person_boot[:,:,None],order_boot].mean(2).reshape(-1,12)
    def effect(row):
        return dict(mean=float(fit[row]@values),ci80=np.quantile(boot@fit[row],[.1,.9]).tolist())
    return dict(visit3_minus_visit1=effect(5),last_minus_first=effect(7),design_rank=8,observations=12)


def main():
    records = np.full((4,2,3,4,3,3),np.nan)
    digest = sha256()
    evidence = []
    refits = []
    for i, p in enumerate(PEOPLE):
        name = f'sub-{p:02d}'
        proof = json.loads(Path(name+'.json').read_text())
        assert proof['relative_rms'] < .01 and len(proof['ranges']) == 9
        with np.load(name+'.npz') as z:
            assert np.array_equal(z['ranks'][:,0], POSITIONS[i])
        evidence.append(proof)
        for group in ['old','cap','a6']:
            path = Path(f'{group}_{p}.json')
            digest.update(path.read_bytes())
            for r in json.loads(path.read_text()):
                v = r['visit']
                seed = r['seed']
                j = (429,1429).index(seed)
                arm = r['arm']
                k = ARMS.index(arm)
                rows = dict(orders(p,v))[seed]
                assert np.array_equal(rows,r['rows'])
                labels = np.array(r['labels'])
                assert labels.shape == (18,) and np.isin(labels,[0,1,2]).all()
                cell = records[i,j,v-1,k]
                assert np.isnan(cell).all()
                for task in range(3):
                    cell[task] = 100*np.bincount(labels[rows//6==task],minlength=3)/6
                if group == 'old':
                    assert r['activity']['degen'] == r['activity']['err'] == 0
                if group == 'cap':
                    refits.append(r['activity']['ft_refits'])
                if (p,v,j) == (8,1,0):
                    smoke, = [s for s in json.loads(Path(group+'_smoke.json').read_text()) if s['arm']==arm]
                    assert smoke['labels'] == r['labels'] and smoke['rows'] == r['rows']
    assert np.isfinite(records).all() and np.allclose(records.sum(-1),100) and len(refits)==24
    total = sum(r['bytes'] for p in evidence for r in p['ranges'])
    assert total < 180_000_000
    out = dict(people=PEOPLE,task_index=['subtraction','memory','lyrics'],label_index=['MI','SUB','WORD'],
               class_rates={},arms={},cap_refits=sum(refits),source_bytes=total,source=evidence,
               prediction_sha256=digest.hexdigest())
    sub = records[...,1]
    effects = sub[...,0] - sub[...,1:].mean(-1)
    for k, arm in enumerate(ARMS):
        contrast = effects[:,:,:,k]
        by_position = {str(pos):np.take_along_axis(contrast, np.argmax(POSITIONS==pos,axis=1)[:,None,None],axis=2)[:,:,0]
                       for pos in (3,4,5)}
        out['class_rates'][arm] = records[:,:,:,k].mean((0,1,2)).tolist()
        out['arms'][arm] = dict(additive=additive(contrast),against_mean=stat(contrast.mean(2)),
                against_memory=stat((sub[:,:,:,k,0]-sub[:,:,:,k,1]).mean(2)),
                against_lyrics=stat((sub[:,:,:,k,0]-sub[:,:,:,k,2]).mean(2)),
                visit=[stat(contrast[:,:,v]) for v in range(3)],
                position={p:stat(c) for p,c in by_position.items()},
                first_minus_last=stat(by_position['3']-by_position['5']))
    for key, i, j in [('cap_minus_old',1,0),('a6_minus_pop',2,3)]:
        out[key] = stat((effects[...,i]-effects[...,j]).mean(2))
    Path('result434.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__ == '__main__':
    main()
