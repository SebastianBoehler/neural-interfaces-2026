"""Use only complete guarded four-person trajectories for paired official scoring."""
import json
from pathlib import Path
import numpy as np
from inputs384 import ARMS,CHAIN,COHORT,FUSION,ORDERS,SEEDS,check,digest,setup
from measure import official


def main():
    setup_data = setup()
    _,_,cfg,meta,rows,_ = setup_data
    import sys
    sys.path.insert(0,str(FUSION))
    from experiment import interval
    proofs,all_outputs = {},{}
    for seed in SEEDS:
        for person in COHORT:
            name = f'{seed}_{person}'
            card = json.loads(Path(name+'.json').read_text())
            check(card['source']==digest('source.sha256') and not card['labels_opened']
                  and card['outputs']==digest(name+'.npz') and card['held']==person and card['seed']==seed
                  and set(card['pair']['source'])==set(card['single']['source'])==set(COHORT)-{person},'fold lineage')
            with np.load(name+'.npz') as z:
                check(set(z.files)=={'index'}|{o+'|'+a for o in ORDERS for a in ARMS}
                      and np.array_equal(z['index'],rows[person]), 'incomplete grid')
                all_outputs[seed,person] = {k:z[k] for k in z.files}
            proofs[name] = card['outputs']
    check(Path('replay.log').read_text().splitlines()[-1]=='OK','historical replay missing')
    Path('before_labels.json').write_text(json.dumps(dict(proofs=proofs,source=digest('source.sha256'),
                                                          scoring_labels_open=False),indent=2))
    metric = official()
    with np.load(CHAIN/'data/proteus_windows.npz') as z:
        y = z['y']
    result = {}
    for seed in SEEDS:
        v = {a:np.empty((4,8)) for a in ARMS}
        change = {a:0 for a in ARMS}
        for i,person in enumerate(COHORT):
            mask = meta['person']==person
            m = all_outputs[seed,person]
            for j,o in enumerate(ORDERS):
                for a in ARMS:
                    pred = m[o+'|'+a]
                    s = metric(y[mask],pred,meta,mask)
                    check(s['n_cells']==2,'cell weights changed')
                    v[a][i,j] = 100*s['balanced_accuracy_cell_mean']
                    change[a] += int(np.sum(pred!=m[o+'|Z']))
        for a,b in [('S','Z'),('S','D'),('S','T'),('S','K'),('S','R'),('R','Z'),('T','Z'),('K','Z'),('D','Z')]:
            v[a+'-'+b] = v[a]-v[b]
        result[seed] = dict(metrics={k:interval(t) for k,t in v.items()},changed_vs_frozen=change)
        check(abs(result[seed]['metrics']['Z']['pp']-81.21561492579204)<1e-9,'stock official endpoint')
    with open('result.json','x') as f:
        json.dump(dict(seeds=result,proof=digest('before_labels.json')),f,indent=2)


if __name__=='__main__':
    main()
