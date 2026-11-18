"""Test reproducible corrections using frozen complete-system decisions."""
import json
import sys
from pathlib import Path
import numpy as np

BAND = Path('/mnt/lustre/work/gehler/gfh527/ni26_band390_v1_20261118')
sys.path.insert(0,str(BAND))
from score import BASE,INPUT,check,digest,setup,summarize

KINDS = ('N','F','A')


def decisions():
    report = json.loads((BAND/'result.json').read_text())
    check(digest(BAND/'before_labels.json')==report['proof'] and
          digest(BAND/'source.sha256')==report['source']=='c6f9f3a964c928a68fb6506081a2e4d0d5541466ee41d5e981d50fab84a5a35f','parent seals')
    seal = json.loads((BAND/'before_labels.json').read_text())
    check(digest(BASE/'before_labels.json')==seal['old'],'broad seals')
    broad = json.loads((BASE/'before_labels.json').read_text())['proofs']
    pred = {}
    with np.load(INPUT/'inputs.npz') as control:
        for oi,o in enumerate(control['orders']):
            names = ['Z']+[k+a for k in KINDS for a in ('1','2','agree','confirmed','opposed')]
            pred[o] = {k:np.full(2655,-1,dtype='int8') for k in names}
            filt = control[o+'|filtered']
            for i in range(oi%2,20,2):
                name = f'core_{i}'
                check(digest(BAND/(name+'.npz'))==seal['new'][name] and digest(BASE/(name+'.npz'))==broad[name],'output hash')
                with np.load(BAND/(name+'.npz')) as new,np.load(BASE/(name+'.npz')) as old:
                    for key in new.files:
                        if not key.startswith(o+'|') or not key.endswith('|rows'):
                            continue
                        rr = new[key]
                        prefix = key.removesuffix('rows')
                        stock = new[prefix+'Z']
                        check(np.array_equal(old[key],rr) and np.array_equal(old[prefix+'Z'],stock)
                              and np.all(pred[o]['Z'][rr]==-1),'stock alignment')
                        pred[o]['Z'][rr] = stock
                        for k in KINDS:
                            db = old if k=='F' else new
                            a,b = (db[prefix+k+seed] for seed in ('386','1386'))
                            same = (a==b) & (a!=stock)
                            favor,oppose = filt[rr,a]>filt[rr,stock],filt[rr,a]<filt[rr,stock]
                            policies = [('1',a),('2',b),('agree',np.where(same,a,stock)),
                                ('confirmed',np.where(same&favor,a,stock)),('opposed',np.where(same&oppose,a,stock))]
                            for label,outputs in policies:
                                check(np.array_equal(outputs[:90],stock[:90]),'prefix')
                                pred[o][k+label][rr] = outputs
            check(all(np.all(x>=0) for x in pred[o].values()),'missing rows')
    return pred


def main():
    pred = decisions()
    out = {o+'|'+k:v for o,p in pred.items() for k,v in p.items()}
    np.savez_compressed('decisions.npz',**out)
    Path('before_labels.json').write_text(json.dumps(dict(outputs=digest('decisions.npz'),owner=digest(BAND/'result.json'),source=digest('source.sha256'))))
    assets,_,_ = setup()
    scorer,_,people,_,meta,_ = assets.load_controls(dtype=np.float32)
    from run import owner
    _,metric = owner()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y = z['y']
    vals = {k:np.zeros((8,10)) for k in pred[scorer.ORDERS[0]]}
    events = {s:{k:np.zeros(4,dtype=int) for k in vals} for s in ('exposed','other')}
    counts = np.zeros(10)
    exposed = np.isin(scorer.PEOPLE,['P01','P02','P03','P09'])
    for i,o in enumerate(scorer.ORDERS):
        for j,person in enumerate(scorer.PEOPLE):
            m = people==person
            for k,out in pred[o].items():
                base = pred[o]['Z'][m]
                hat,gold = out[m],y[m]
                s = metric(gold,hat,meta,m)
                values = [np.sum(hat!=base),np.sum((hat==gold)&(hat!=base)),
                          np.sum((base==gold)&(hat!=base)),np.sum((hat!=base)&(hat!=gold)&(base!=gold))]
                events['exposed' if exposed[j] else 'other'][k] += values
                vals[k][i,j] = 100*s['balanced_accuracy_cell_mean']
                counts[j] = s['n_cells']
    for k in KINDS:
        for a in ('agree','confirmed','opposed'):
            vals[k+a+'-Z'] = vals[k+a]-vals['Z']
        vals[k+'agree-mean'] = vals[k+'agree']-(vals[k+'1']+vals[k+'2'])/2
    vals['Nconfirmed-Fconfirmed'] = vals['Nconfirmed']-vals['Fconfirmed']
    groups = {n:{k:summarize(v[:,m],counts[m]) for k,v in vals.items()} for n,m in
              [('all',np.ones(10,dtype=bool)),('exposed',exposed),('other',~exposed)]}
    check(abs(groups['all']['Z']['official']-76.27933502145758)<1e-8,'stock')
    Path('result.json').write_text(json.dumps(dict(groups=groups,people=list(scorer.PEOPLE),
        events={g:{k:v.tolist() for k,v in d.items()} for g,d in events.items()},
        proof=digest('before_labels.json')),indent=2))


if __name__=='__main__':
    main()
