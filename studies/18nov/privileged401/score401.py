"""Use fixed official row predictions for whole, full, error and interface comparisons."""
import json
import sys
from pathlib import Path
import numpy as np
from full401 import SRC,BASE,ARMS,HELD,INPUT,check,digest,setup,members


def main():
    card=members()
    check(Path('replay.log').read_text().splitlines()[-1]=='OK','v3 guard')
    assets,_,_=setup()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        meta={k:z[k] for k in ('subject','session','interface')}
    meta['person']=np.array([p.rsplit('/',1)[-1] for p in meta['subject']])
    meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    with np.load(SRC/'members.npz') as z:
        rows=z['index']
        whole={a:z[a].argmax(1) for a in ARMS}
    with np.load(INPUT/'inputs.npz') as z:
        orders=z['orders'].tolist()
    full={}
    seals={}
    for j,o in enumerate(orders):
        n=f'order_{j}'
        r=json.loads(Path(n+'.json').read_text())
        check(r==dict(source=digest('source.sha256'),members=card['members'],order=o,outputs=digest(n+'.npz')),'actor seal')
        seals[n]=r
        with np.load(n+'.npz') as z:
            check(set(z.files)==set(('index','Z')+ARMS) and np.array_equal(rows,z['index']),'held grid')
            full[o]={a:z[a] for a in ('Z',)+ARMS}
    Path('before_labels.json').write_text(json.dumps(dict(shards=seals,members=card),indent=2))
    from run import owner
    _,official=owner()
    sys.path.insert(0,str(BASE/'ni26_fusion383_v1_20261118'))
    from experiment import interval
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y=z['y']
    report={}
    for family,arms in [('whole',ARMS),('full',('Z',)+ARMS)]:
        values={a:np.zeros((4,8)) for a in arms}
        cognitive={f:{a:np.zeros((4,8)) for a in arms} for f in ('Graz','BrainHero')}
        errors={a:np.zeros((3,3),dtype=int) for a in arms}
        for i,p in enumerate(HELD):
            mm=meta['person']==p
            pos=mm[rows]
            for j,o in enumerate(orders):
                for a in arms:
                    v=(whole[a] if family=='whole' else full[o][a])[pos]
                    m=official(y[mm],v,meta,mm)
                    check(m['n_cells']==2,'cell count')
                    values[a][i,j]=100*m['balanced_accuracy_cell_mean']
                    np.add.at(errors[a],(y[mm],v),1)
                    for f in cognitive:
                        iface=meta['interface'][mm]==f
                        cognitive[f][a][i,j]=100*np.mean([np.mean(v[iface&(y[mm]==label)]==label) for label in (1,2)])
        pairs=[('X',b) for b in arms if b!='X']
        for a,b in pairs:
            values[a+'-'+b]=values[a]-values[b]
        report[family]=dict(metrics={k:interval(v) for k,v in values.items()},
            cognitive={f:{a+'-'+b:interval(v[a]-v[b]) for a,b in pairs} for f,v in cognitive.items()},
            confusion={k:v.tolist() for k,v in errors.items()})
    check(abs(report['full']['metrics']['Z']['pp']-81.21561492579204)<1e-8,'stock metric')
    Path('result.json').write_text(json.dumps(dict(results=report,source=card,proof=digest('before_labels.json')),indent=2))


if __name__=='__main__':
    main()
