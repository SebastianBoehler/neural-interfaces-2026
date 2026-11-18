"""Locate how changing only rebalancing state propagated through EM and TSEM."""
import json
import numpy as np
from score import HERE,ORIG,sc,summary_factory
from cross import TRACE


def main():
    with np.load(ORIG/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        y,cell,person=(z[k+'|all_loso'] for k in ('y','cell','person'))
    summary=summary_factory(np.array([len(np.unique(cell[person==p])) for p in sc.PEOPLE]))
    out={}
    for label,bases in [('B2_pool',sc.BASES[:1]),('six_bases',sc.BASES)]:
        out[label]={}
        for stage in ('mix','rebal','em','V6Fr'):
            values={a:np.empty((8,10)) for a in ('RB','L','D','crossL','crossD')}
            for a,v in values.items():
                for i,o in enumerate(sc.ORDERS):
                    with np.load((HERE if a.startswith('cross') else TRACE)/f'{a}_{o}.npz') as z:
                        v[i]=np.mean([sc.person_accuracy(z[b+'|'+stage],y,cell,person,sc.PEOPLE) for b in bases],axis=0)
            out[label][stage]={a+'-'+b:summary(values[a]-values[b]) for a,b in
                               [('crossL','L'),('crossL','RB'),('crossL','crossD')]}
            if stage=='mix':
                assert np.array_equal(values['crossL'],values['L']) and np.array_equal(values['crossD'],values['D'])
    (HERE/'cross_stages.json').write_text(json.dumps(out))


if __name__=='__main__':
    main()
