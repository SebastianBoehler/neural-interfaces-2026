"""Evaluate pinned canonical predictions using the four-person official measure."""
import sys
import json
from pathlib import Path
import numpy as np
from rank394 import PEOPLE,SEEDS,ARMS,ROOT,INPUT,check,digest,setup


def main():
    check(Path('replay.log').read_text().splitlines()[-1]=='OK','v3 guard')
    assets,_,_=setup()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        meta={k:z[k] for k in ('subject','session','interface')}
    meta['person']=np.array([k.rsplit('/',1)[-1] for k in meta['subject']])
    meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    with np.load(INPUT/'inputs.npz') as z:
        orders=z['orders'].tolist()
    data,seals={},{}
    for s in SEEDS:
        for p in PEOPLE:
            n=f'{s}_{p}'
            r=json.loads(Path(n+'.json').read_text())
            check(r==dict(source=digest('source.sha256'),outputs=digest(n+'.npz')),'fit seals')
            with np.load(n+'.npz') as z:
                check(np.array_equal(z['index'],np.flatnonzero(meta['person']==p))
                      and set(z.files)=={'index'}|{o+'|'+a for o in orders for a in ARMS},'rows')
                data[s,p]={k:z[k] for k in z.files}
            seals[n]=r
    Path('before_labels.json').write_text(json.dumps(seals,indent=2))
    from run import owner
    _,metric=owner()
    sys.path.insert(0,str(ROOT/'ni26_fusion383_v1_20261118'))
    from experiment import interval
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y=z['y']
    out={}
    for s in SEEDS:
        vals={a:np.zeros((4,8)) for a in ARMS}
        for i,p in enumerate(PEOPLE):
            m=meta['person']==p
            for j,o in enumerate(orders):
                for a in ARMS:
                    sc=metric(y[m],data[s,p][o+'|'+a],meta,m)
                    check(sc['n_cells']==2,'cell weights')
                    vals[a][i,j]=100*sc['balanced_accuracy_cell_mean']
        pairs=[('JT',b) for b in ('JP','JU','N','D','Z')]+[('KT',b) for b in ('KP','KU','N','Z')]
        for a,b in pairs:
            vals[a+'-'+b]=vals[a]-vals[b]
        out[s]={k:interval(v) for k,v in vals.items()}
        check(abs(out[s]['Z']['pp']-81.21561492579204)<1e-9,'cap zero')
    Path('result.json').write_text(json.dumps(dict(seeds=out,seal=digest('before_labels.json')),indent=2))


if __name__=='__main__':
    main()
