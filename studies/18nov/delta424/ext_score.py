"""Pair external selectivity within the same people and pinned orders."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from external import HIST
from fits import SEEDS, ARMS, WORK, sha

spec = importlib.util.spec_from_file_location('metrics418',HIST/'score.py')
metric = importlib.util.module_from_spec(spec)
assert sha(HIST/'score.py')=='cabbb967cfef24fcce27d57807c8089b9abaaca71c1e1a4f264c153756fced9d'
spec.loader.exec_module(metric)


def main():
    assert sha(HIST/'result.json')=='bdfd662917f2d729b55926e6d1d98478f81f5de3bbd102f4e0e7745aa438994b'
    old_result = json.loads((HIST/'result.json').read_text())
    package = WORK/'ni26_select415_20261118'
    commitments = {p:sha(f'ext{p}.npz') for p in range(1,27)}
    Path('ext_before_labels.json').write_text(json.dumps(commitments,indent=2))
    with np.load(WORK/'neural-interfaces-2026/data/shin2018/windows.npz') as z:
        labels,runs = z['y'],z['run']
    rows = {}
    for p in commitments:
        per = {}
        with np.load(f'ext{p}.npz') as z, np.load(HIST/f'person{p}.npz') as old:
            for o in (415,1415):
                idx = z[str(o)]
                assert np.array_equal(idx,old[str(o)])
                preds = {f'{a}{s}':z[f'{o}|{a}{s}'] for s in SEEDS for a in ARMS}
                for a in ('native','pop','J382','J1382','K382','K1382'):
                    preds[a] = old[f'{o}|{a}']
                for a in ('old','cap'):
                    bank=json.loads((package/f'{a}_{6+(p-1)//5}.json').read_text())
                    [r]=[r for r in bank if r['seed']==o and r['person']==p]
                    assert np.array_equal(idx,r['raw'])
                    preds[a]=np.array(r['labels'])
                for a,v in preds.items():
                    per.setdefault(a,[]).append(metric.selectivity(v,labels[idx],runs[idx]))
        for a,v in per.items():
            rows.setdefault(a,[]).append(np.mean(v,axis=0))
    rows={a:np.array(v) for a,v in rows.items()}
    for a in ('native','pop','J382','J1382','K382','K1382','old','cap'):
        np.testing.assert_allclose(rows[a].mean(0),old_result['arms'][a],atol=1e-10,rtol=0)
    draw=np.random.default_rng(418).integers(26,size=(10000,26))
    out={}
    for s in SEEDS:
        t=rows[f'T{s}']
        pairs={a:t[:,0]-rows[a][:,0] for a in (f'S{s}',f'C{s}',f'blank{s}','native',f'J{s}',f'K{s}')}
        for a in ('old','cap'):
            assert rows[a].shape == (26,)
            pairs[a] = t[:,0]-rows[a]
        pairs['horizon']=t[:,0]-t[:,1]
        out[s]={a:metric.summary(v,draw) for a,v in pairs.items()}
    Path('ext_result.json').write_text(json.dumps(dict(arms={a:v.mean(0).tolist() for a,v in rows.items()},
        T_minus=out,people=26,orders=[415,1415],fits=sha('fits.json'),commitments=sha('ext_before_labels.json')),indent=2))


if __name__=='__main__':
    main()
