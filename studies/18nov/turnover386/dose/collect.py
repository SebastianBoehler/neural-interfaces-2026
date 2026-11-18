"""Score both preregistered doses and every constituent at the same trial rows."""
import json
from pathlib import Path
import numpy as np
from dose import ARMS,INPUT,check,digest,setup,summarize


def main():
    arms = ARMS
    assets,_,_ = setup()
    proofs,payload = {},{}
    source = digest('source.sha256')
    for i in range(40):
        name = f'run_{i}'
        proof = json.loads(Path(name+'.json').read_text())
        check(proof==dict(source=source,outputs=digest(name+'.npz')),'shard integrity')
        proofs[name] = proof['outputs']
        with np.load(name+'.npz') as z:
            check(not payload.keys()&set(z.files),'duplicates')
            payload.update({k:z[k] for k in z.files})
    check(len(payload)==2*8*14*(1+len(arms)),'incomplete endpoint')
    check(Path('replay.log').read_text().splitlines()[-1]=='OK','v3 replay')
    Path('before_labels.json').write_text(json.dumps(dict(proofs=proofs,source=source),indent=2))
    scorer,_,people,_,meta,_ = assets.load_controls(dtype=np.float32)
    from run import owner
    _,metric = owner()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y = z['y']
    vals = {f'{d}_{a}':np.zeros((8,10)) for d in (16,4) for a in arms}
    counts = np.zeros(10)
    with np.load(INPUT/'inputs.npz') as z:
        for i,o in enumerate(scorer.ORDERS):
            for den in (16,4):
                pred = {a:np.full(len(y),-1) for a in arms}
                for sid,rr in enumerate(np.split(z[o+'|index'],z[o+'|bounds'][1:-1])):
                    key = f'{den}|{o}|{sid}|'
                    check(np.array_equal(payload[key+'rows'],rr),'order')
                    for a in arms:
                        pred[a][rr] = payload[key+a]
                for j,p in enumerate(scorer.PEOPLE):
                    mask = people==p
                    for a in arms:
                        score = metric(y[mask],pred[a][mask],meta,mask)
                        counts[j] = score['n_cells']
                        vals[f'{den}_{a}'][i,j] = 100*score['balanced_accuracy_cell_mean']
    for den in (16,4):
        for a,b in [(a,'Z') for a in arms[1:]]+[('Fpair','Ppair'),('Fpair','B')]:
            vals[f'{den}_{a}-{b}'] = vals[f'{den}_{a}']-vals[f'{den}_{b}']
    vals['high-low_Fpair'] = vals['4_Fpair']-vals['16_Fpair']
    mask = np.isin(scorer.PEOPLE,['P01','P02','P03','P09'])
    result = {name:{k:summarize(v[:,ix],counts[ix]) for k,v in vals.items()}
              for name,ix in [('all',np.ones(10,dtype=bool)),('exposed',mask),('other',~mask)]}
    for den in (16,4):
        check(abs(result['all'][f'{den}_Z']['official']-76.27933502145758)<1e-8,'stock score')
    Path('result.json').write_text(json.dumps(dict(groups=result,people=list(scorer.PEOPLE),
        source=source,proof=digest('before_labels.json')),indent=2))


if __name__=='__main__':
    main()
