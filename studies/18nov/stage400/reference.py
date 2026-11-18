"""Paired complete endpoints against a separately selected complete JS stack."""
import json
import numpy as np
from score import HERE,ORIG,sc,digest,summary_factory
from capture import WORK


def main():
    path=WORK/'ni26_joint373_v2_20261117/controls.npz'
    assert digest(path)=='3f5ceae9662d759112352701c55a194ccdb78249e171ec2d97a604c9aba88b2b'
    with np.load(ORIG/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        y,cell,person=(z[k+'|all_loso'] for k in ('y','cell','person'))
    summary=summary_factory(np.array([len(np.unique(cell[person==p])) for p in sc.PEOPLE]))
    result={a:np.empty((8,10)) for a in ('crossL','crossD','JS','frozen')}
    with np.load(path) as control:
        assert np.array_equal(person,[p.split('/')[-1] for p in control['subject']])
        for i,order in enumerate(sc.ORDERS):
            for name,values in result.items():
                if name.startswith('cross'):
                    with np.load(HERE/f'final_{name}_{order}_B2_pool.npz') as z:
                        out=z['scores']
                else:
                    out=np.eye(3)[control[f'{name}_{order}']]
                values[i]=sc.person_accuracy(out,y,cell,person,sc.PEOPLE)
    assert abs(summary(result['JS'])['mean']-76.55173151211109)<1e-9
    assert abs(summary(result['frozen'])['mean']-76.27933502145761)<1e-9
    out={a+'-JS':summary(result[a]-result['JS']) for a in ('crossL','crossD')}
    (HERE/'reference.json').write_text(json.dumps(out,indent=1))


if __name__=='__main__':
    main()
