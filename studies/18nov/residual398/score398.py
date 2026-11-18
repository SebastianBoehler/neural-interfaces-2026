"""Seal alternative decoder and complete CP-consumer decisions before opening gold."""
import json
import sys
from pathlib import Path
import numpy as np
from full398 import ARMS,INPUT,SEEDS,PEOPLE,ROOT,digest,check,setup

WHOLE=('RT','RP','RU','CT','A6','JS')


def main():
    check(Path('replay.log').read_text().splitlines()[-1]=='OK','v3 replay')
    assets,_,_=setup()
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        meta={k:z[k] for k in ('subject','session','interface')}
    meta['person']=np.array([v.rsplit('/',1)[-1] for v in meta['subject']])
    meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    with np.load(INPUT/'inputs.npz') as z:
        orders=z['orders'].tolist()
    data,seals={},{}
    for p in PEOPLE:
        rows=np.flatnonzero(meta['person']==p)
        actor=json.loads(Path(p+'.json').read_text())
        check(actor['source']==digest('source.sha256') and actor['outputs']==digest(p+'.npz')
              and actor['held']==p and not actor['held_labels_opened'] and actor['fits']==digest('fits.json'),'actor')
        seals[p]=actor
        with np.load(p+'.npz') as bank:
            check(np.array_equal(bank['index'],rows),'actor rows')
            for s in SEEDS:
                n=f'full_{s}_{p}'
                r=json.loads(Path(n+'.json').read_text())
                check(r==dict(source=digest('source.sha256'),outputs=digest(n+'.npz')),'CP shard')
                seals[n]=r
                with np.load(n+'.npz') as z:
                    check(np.array_equal(z['index'],rows) and set(z.files)=={'index'}|{o+'|'+a for o in orders for a in ARMS},'full rows')
                    data[s,p,'full']={(a,o):z[o+'|'+a] for o in orders for a in ARMS}
                data[s,p,'whole']={(a,o):bank[(a if a in ('A6','JS') else f'{s}|{a}')+'|'+o].argmax(1)
                                     for o in orders for a in WHOLE}
    Path('before_labels.json').write_text(json.dumps(seals,indent=2))
    from run import owner
    _,metric=owner()
    sys.path.insert(0,str(ROOT/'ni26_fusion383_v1_20261118'))
    from experiment import interval
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y=z['y']
    out={}
    for s in SEEDS:
        out[s]={}
        for family,arms in [('whole',WHOLE),('full',ARMS)]:
            vals={a:np.zeros((4,8)) for a in arms}
            for i,p in enumerate(PEOPLE):
                mask=meta['person']==p
                for j,o in enumerate(orders):
                    for a in arms:
                        m=metric(y[mask],data[s,p,family][a,o],meta,mask)
                        check(m['n_cells']==2,'cell weight')
                        vals[a][i,j]=100*m['balanced_accuracy_cell_mean']
            for a in arms:
                if a!='RT':
                    vals['RT-'+a]=vals['RT']-vals[a]
            summary={k:interval(v) for k,v in vals.items()}
            if family=='whole':
                check(abs(summary['A6']['pp']-60.78519278107362)<1e-8 and abs(summary['JS']['pp']-81.23552194696784)<1e-8,'whole controls')
            else:
                check(abs(summary['Z']['pp']-81.21561492579204)<1e-8,'CP control')
            out[s][family]=summary
    Path('result.json').write_text(json.dumps(dict(seeds=out,proof=digest('before_labels.json')),indent=2))


if __name__=='__main__':
    main()
