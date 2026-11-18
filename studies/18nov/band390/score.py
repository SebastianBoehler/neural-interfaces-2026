"""Score matched archived broad swaps alongside newly localized trajectories."""
import json
from pathlib import Path
import numpy as np
from worker import CORE,GEOM,INPUT,check,digest,setup
from score386 import summarize

BASE = Path('/mnt/lustre/work/gehler/gfh527/ni26_turn386_v1_20261118')
REUSE = ('B','F386','F1386')


def roster():
    check(digest(BASE/'result.json')=='aaf5795a46496e729efe3766182b57b52065fc936d1e1d01dc17f9671cc63cf8','prior result')
    report = json.loads((BASE/'result.json').read_text())
    check(digest(BASE/'before_labels.json')==report['proof'],'prior endpoint seals')
    old = json.loads((BASE/'before_labels.json').read_text())['proofs']
    proof,payload = {},{}
    for i in range(20):
        for mode in ('core','geometry'):
            name = f'{mode}_{i}'
            h = digest(name+'.npz')
            check(json.loads(Path(name+'.json').read_text())==dict(source=digest('source.sha256'),outputs=h),'shard lineage')
            proof[name] = h
            with np.load(name+'.npz') as z:
                for k in z.files:
                    if k in payload:
                        check(k.endswith('|rows') and np.array_equal(z[k],payload[k]),'duplicate key')
                    payload[k] = z[k]
            if mode=='core':
                path = BASE/(name+'.npz')
                check(digest(path)==old[name],'prior shard hash')
                with np.load(path) as z:
                    for k in z.files:
                        field = k.split('|')[-1]
                        if field in ('rows','Z','Z_trace'):
                            check(np.array_equal(z[k],payload[k]),'same held rows and stock')
                        if field.removesuffix('_trace') in REUSE:
                            payload[k] = z[k]
    check(len(payload)==112*(1+2*len(CORE+GEOM+REUSE)),'incomplete grid')
    Path('before_labels.json').write_text(json.dumps(dict(new=proof,old=report['proof'],source=digest('source.sha256')),indent=2))
    return payload


def main():
    check(Path('replay.log').read_text().splitlines()[-1]=='OK','v3 replay')
    payload = roster()
    assets,_,_ = setup()
    scorer,_,people,_,meta,_ = assets.load_controls(dtype=np.float32)
    from run import owner
    _,metric = owner()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y = z['y']
    arms = CORE+GEOM+REUSE
    values = {a:np.zeros((8,10)) for a in arms}
    counts = np.zeros(10)
    with np.load(INPUT/'inputs.npz') as z:
        for i,o in enumerate(scorer.ORDERS):
            pred = {a:np.full(len(y),-1) for a in arms}
            for sid,rr in enumerate(np.split(z[o+'|index'],z[o+'|bounds'][1:-1])):
                key = f'{o}|{sid}|'
                check(np.array_equal(rr,payload[key+'rows']),'order identity')
                budgets = [payload[key+a+'_trace'][0] for a in arms if a!='Z']
                check(len(set(budgets))==1,'different exchange dose')
                for a in arms:
                    pred[a][rr] = payload[key+a]
            for j,person in enumerate(scorer.PEOPLE):
                mask = people==person
                for a in arms:
                    m = metric(y[mask],pred[a][mask],meta,mask)
                    values[a][i,j] = 100*m['balanced_accuracy_cell_mean']
                    counts[j] = m['n_cells']
    contrasts = [(a,'Z') for a in arms if a!='Z']+[('G','H'),('G','R')]
    for seed in ('386','1386'):
        contrasts.extend(('N'+seed,b) for b in ('A'+seed,'F'+seed,'B'))
    for a,b in contrasts:
        values[a+'-'+b] = values[a]-values[b]
    old = np.isin(scorer.PEOPLE,['P01','P02','P03','P09'])
    groups = {name:{k:summarize(v[:,mask],counts[mask]) for k,v in values.items()}
              for name,mask in [('all',np.ones(10,dtype=bool)),('other',~old),('exposed',old)]}
    check(abs(groups['all']['Z']['official']-76.27933502145758)<1e-8,'stock metric')
    stops = assets.returning_guard(scorer.ORDERS,90)
    Path('result.json').write_text(json.dumps(dict(groups=groups,people=list(scorer.PEOPLE),
        returning_stops=stops,source=digest('source.sha256'),proof=digest('before_labels.json')),indent=2))


if __name__=='__main__':
    main()
