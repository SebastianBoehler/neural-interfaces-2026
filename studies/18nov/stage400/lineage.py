"""Track each original gain and injury into mixture and the full personalized decision."""
import json
import numpy as np
from capture import HERE,ORIG
from score import sc,summary_factory


def main():
    with np.load(ORIG/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        y,cell,people=(z[k+'|all_loso'] for k in ('y','cell','person'))
    count=np.array([len(np.unique(cell[people==p])) for p in sc.PEOPLE])
    summarize=summary_factory(count)
    def mass(mask):
        by_person=[]
        for p in sc.PEOPLE:
            by_cell=[]
            for c in np.unique(cell[people==p]):
                by_cell.append(np.mean([mask[(cell==c)&(y==k)].mean() for k in np.unique(y[cell==c])]))
            by_person.append(100*np.mean(by_cell))
        return by_person
    all_values={}
    for oi,o in enumerate(sc.ORDERS):
        hit={}
        for a in ('L','RB','D'):
            with np.load(ORIG/f'runs/representation_streams/ttt18{o}_n395{a.lower()}_base-all_loso-ensemble2.npz') as z:
                hit[a,'member']=(z['proba'].argmax(1)==y).astype(int)
            with np.load(HERE/f'{a}_{o}.npz') as z:
                hit[a,'mix']=(z['B2_pool|mix'].argmax(1)==y).astype(int)
            with np.load(ORIG/f'runs/s281_v66_cp/{a}_{o}_B2_pool.npz') as z:
                hit[a,'CP']=(z['scores'].argmax(1)==y).astype(int)
        for b in ('RB','D'):
            for stage in ('mix','CP'):
                start=hit['L','member']-hit[b,'member']
                end=hit['L',stage]-hit[b,stage]
                for old in (-1,0,1):
                    for new in (-1,0,1):
                        k=f'L-{b}|{stage}|{old}:{new}'
                        all_values.setdefault(k,np.zeros((8,10)))[oi]=mass((start==old)&(end==new))
    result={k:summarize(v)['mean'] for k,v in all_values.items()}
    for prefix in ('L-RB|mix','L-RB|CP','L-D|mix','L-D|CP'):
        assert abs(sum(v for k,v in result.items() if k.startswith(prefix+'|'))-100)<1e-9
    (HERE/'lineage.json').write_text(json.dumps(result,indent=1))


if __name__=='__main__':
    main()
