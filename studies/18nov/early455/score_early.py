"""Score locked early-readout predictions and count possible corrections on their disagreements."""
import json
from pathlib import Path
import numpy as np
import score379 as ref
from fusion_inputs import official
from score import interval

PRODUCER='b22520ce11d59f8ba4d8e4780a42028e6b432be961628318f986ef1f69e9d5a3'
proofs={}
banks={}
for o in ref.ORDERS:
    path=Path(o+'.npz')
    proofs[o]=json.loads(path.with_suffix('.json').read_text())
    assert proofs[o]['output']==ref.sha(path) and proofs[o]['source']==PRODUCER
    with np.load(path) as z:
        banks[o]=dict(z)
dfile=ref.BASE/'data/proteus_windows.npz'
assert ref.sha(dfile)==ref.DATA_SHA
with np.load(dfile) as z:
    y=z['y']
    meta={k:z[k] for k in ('subject','session','interface')}
meta['stream']=np.char.add(np.char.add(meta['subject'],'|'),meta['session'].astype(str))
people=np.array([s.rsplit('/',1)[-1] for s in meta['subject']])
metric=official()
counts=[]
scores={a:np.empty((10,8)) for a in ('stock','adaptive','fixed')}
for i,p in enumerate(ref.PEOPLE):
    mask=people==p
    counts.append(len(set(zip(meta['session'][mask],meta['interface'][mask]))))
    for j,o in enumerate(ref.ORDERS):
        for a in scores:
            m=metric(y[mask],banks[o][a][mask],meta,mask)
            assert m['n_cells']==counts[-1]
            scores[a][i,j]=100*m['balanced_accuracy_cell_mean']
counts=np.array(counts)
absolute={a:interval(v,counts)['pp'] for a,v in scores.items()}
assert counts.sum()==28 and abs(absolute['stock']-76.1039)<.0001
effects={}
for a,c in (('adaptive','stock'),('adaptive','fixed'),('fixed','stock')):
    delta=scores[a]-scores[c]
    correction=regression=changed=0
    for b in banks.values():
        correction+=int(((b[a]==y)&(b[c]!=y)).sum())
        regression+=int(((b[a]!=y)&(b[c]==y)).sum())
        changed+=int((b[a]!=b[c]).sum())
        assert np.all(b[a][~b['eligible']]==b[c][~b['eligible']])
    effects[a+'-'+c]=dict(**interval(delta,counts),orders=dict(zip(ref.ORDERS,np.average(delta,axis=0,weights=counts))),
        changed=changed,corrections=correction,regressions=regression)
Path('result.json').write_text(json.dumps(dict(absolute=absolute,effects=effects,proofs=proofs,scorer=ref.sha(Path('source.sha256'))),indent=2))
