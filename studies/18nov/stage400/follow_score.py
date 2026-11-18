"""Compare the completed blend intervention to its source donor and archived complete controls."""
import json
import numpy as np
from score import sc,summary_factory,digest
from capture import HERE,ORIG
from cross import TRACE


def main():
    assert (TRACE/'replay.log').read_text().splitlines()[-1]=='OK'
    hist=json.loads((ORIG.parent/'result.json').read_text())
    with np.load(ORIG/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        y,cell,person=(z[k+'|all_loso'] for k in ('y','cell','person'))
    counts=np.array([len(np.unique(cell[person==p])) for p in sc.PEOPLE])
    summary=summary_factory(counts)
    arms=('control','RB','L','D','crossL','crossD')
    vals=np.empty((6,6,8,10))
    for oi,order in enumerate(sc.ORDERS):
        for bi,base in enumerate(sc.BASES):
            for ai,arm in enumerate(arms):
                p=(HERE/f'final_{arm}_{order}_{base}.npz' if arm.startswith('cross') else
                   ORIG/f'runs/s281_v66_cp/{arm}_{order}_{base}.npz')
                with np.load(p) as z:
                    v=z['scores']
                vals[ai,bi,oi]=sc.person_accuracy(v,y,cell,person,sc.PEOPLE)
    result={}
    for label,ids in [('B2_pool',[0]),('six_bases',list(range(6)))]:
        v=dict(zip(arms,vals[:,ids].mean(1),strict=True))
        result[label]={a:summary(w) for a,w in v.items()}
        for a in arms[:4]:
            assert abs(result[label][a]['mean']-hist[label][a]['official']['mean'])<1e-9
        for a,b in [('crossL','crossD'),('crossL','L'),('crossL','RB'),('crossD','D'),('crossL','control')]:
            result[label][a+'-'+b]=summary(v[a]-v[b])
    result['source_sha256']=digest(HERE/'source.sha256')
    (HERE/'result.json').write_text(json.dumps(result,indent=1))


if __name__=='__main__':
    main()
