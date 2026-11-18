"""Attribute confidence and historical pseudo-label correctness after locking endpoints."""
import json
from pathlib import Path
import importlib.util
import numpy as np
from score import BASE
from selector import exchange
from worker import CORE,GEOM,INPUT,SCHEDULE,check,digest,setup


def main():
    result = json.loads(Path('result.json').read_text())
    check(digest('before_labels.json')==result['proof'],'unsealed endpoint')
    spec = importlib.util.spec_from_file_location('prior_selector',BASE/'selector.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    assets,cp,cfg = setup()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y = z['y']
        people = np.array([v.rsplit('/',1)[-1] for v in z['subject']])
    stage = cp(cfg)
    feat = np.load(SCHEDULE/'features.npy',mmap_mode='r')
    arms = CORE[1:]+GEOM+('B','F386','F1386')
    totals = {g:{a:np.zeros(5) for a in arms} for g in ('exposed','other')}
    with np.load(INPUT/'inputs.npz') as z:
        for o in z['orders']:
            scores = z[o+'|raw']
            for rr in np.split(z[o+'|index'],z[o+'|bounds'][1:-1]):
                g = 'exposed' if people[rr[0]] in ('P01','P02','P03','P09') else 'other'
                features = np.log(np.maximum(np.diagonal(feat[rr],axis1=2,axis2=3),1e-12)).reshape(len(rr),-1)
                for t in range(20,len(rr),20):
                    raw = scores[rr[:t]]
                    c,labels = raw.max(1),raw.argmax(1)
                    stock = stage.select(labels,c)
                    good = labels==y[rr[:t]]
                    for a in arms:
                        method = prior.exchange if a.startswith(('B','F')) else exchange
                        active = method(labels,c,stock,a,features[:t])
                        added,removed = np.setdiff1d(active,stock),np.setdiff1d(stock,active)
                        totals[g][a] += [len(added),c[added].sum(),c[removed].sum(),good[added].sum(),good[removed].sum()]
    out = {}
    for g,table in totals.items():
        out[g] = {}
        for a,(n,ac,rc,ag,rg) in table.items():
            out[g][a] = dict(swaps=int(n),added_conf=ac/n,removed_conf=rc/n,added_correct=ag/n,removed_correct=rg/n)
            check(sum(totals[z][a][0] for z in totals)==6593,'reconstructed dose')
    Path('diagnostic.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
