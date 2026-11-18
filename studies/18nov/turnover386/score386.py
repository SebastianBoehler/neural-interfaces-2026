"""Paired cell- and person-balanced complete-system replay and subset intervals."""
import json
from pathlib import Path
import numpy as np
from worker import CORE,GEOM,INPUT,check,digest,setup


def summarize(v,counts):
    groups = ([0,1,2,3],[4,5],[6,7])
    rng = np.random.default_rng(386)
    people = rng.integers(v.shape[1],size=(4000,v.shape[1]))
    d = np.mean([v[rng.choice(g,size=(4000,len(g))),:].mean(1) for g in groups],axis=0)
    drawn = np.take_along_axis(d,people,axis=1)
    weights = counts[people]
    est = np.mean([v[g,:].mean(0) for g in groups],axis=0)
    return dict(official=float(np.average(est,weights=counts)),
                interval=np.percentile((drawn*weights).sum(1)/weights.sum(1),[10,90]).tolist(),
                people=float(est.mean()),people_interval=np.percentile(drawn.mean(1),[10,90]).tolist(),
                by_person=est.tolist())


def main():
    assets,_,_ = setup()
    arms = CORE+GEOM
    names = [f'{mode}_{i}' for i in range(20) for mode in ('core','geometry')]
    source = digest('source.sha256')
    proofs = {}
    payload = {}
    check(Path('replay.log').read_text().splitlines()[-1]=='OK','v3 replay')
    for name in names:
        card = json.loads(Path(name+'.json').read_text())
        check(card==dict(source=source,outputs=digest(name+'.npz')),'worker lineage')
        proofs[name] = card['outputs']
        with np.load(name+'.npz') as z:
            for k in z.files:
                if k in payload:
                    check(k.endswith('|rows') and np.array_equal(z[k],payload[k]),'shard collision')
                payload[k]=z[k]
    check(len(payload)==8*14*(1+2*len(arms)),'incomplete grid')
    Path('before_labels.json').write_text(json.dumps(dict(source=source,proofs=proofs),indent=2))
    scorer,_,people,_,meta,_ = assets.load_controls(dtype=np.float32)
    from run import owner
    _,metric = owner()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y = z['y']
    vals = {arm:np.zeros((8,10)) for arm in arms}
    trace = {arm:np.zeros(3,dtype='int64') for arm in arms}
    counts = np.zeros(10)
    with np.load(INPUT/'inputs.npz') as z:
        check(list(scorer.ORDERS)==list(z['orders'])==['within0','within1','within2','within3','full0','full1','runs0','runs1'],'order groups')
        for i,order in enumerate(scorer.ORDERS):
            pred = {a:np.full(len(y),-1) for a in arms}
            for sid,rr in enumerate(np.split(z[order+'|index'],z[order+'|bounds'][1:-1])):
                key = f'{order}|{sid}|'
                check(np.array_equal(rr,payload[key+'rows']),'order mismatch')
                for arm in arms:
                    pred[arm][rr] = payload[key+arm]
                    trace[arm] += payload[key+arm+'_trace']
            for j,p in enumerate(scorer.PEOPLE):
                mask = people==p
                for arm in arms:
                    m = metric(y[mask],pred[arm][mask],meta,mask)
                    vals[arm][i,j] = 100*m['balanced_accuracy_cell_mean']
                    counts[j] = m['n_cells']
    bases = [(a,'Z') for a in arms[1:]]+[('G','H')]
    for s in ('386','1386'):
        bases += [('F'+s,b) for b in ('P'+s,'B')]+[('G','F'+s)]
    for a,b in bases:
        vals[a+'-'+b] = vals[a]-vals[b]
    exposed = np.isin(scorer.PEOPLE,['P01','P02','P03','P09'])
    results = {g:{a:summarize(v[:,ix],counts[ix]) for a,v in vals.items()}
               for g,ix in [('all',np.ones(10,dtype=bool)),('exposed',exposed),('other',~exposed)]}
    check(abs(results['all']['Z']['official']-76.27933502145758)<1e-8,'stock metric')
    check(len(set(x[0] for a,x in trace.items() if a!='Z'))==1,'different arm doses')
    Path('result.json').write_text(json.dumps(dict(groups=results,people=list(scorer.PEOPLE),
        trace={a:x.tolist() for a,x in trace.items()},source=source,proof=digest('before_labels.json')),indent=2))


if __name__=='__main__':
    main()
