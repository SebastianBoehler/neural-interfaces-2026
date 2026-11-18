"""Score literal treatment trajectories with matched original person/order bootstrap draws."""
import hashlib
import json
import numpy as np
from capture import HERE,ORIG,RUN
import sys
sys.path.insert(0,str(RUN))
import score_s_v66p as sc

ARMS=('control','RB','L','M','D')
STAGES=('member','mix','rebal','em','V6Fr','count','CP')
PAIRS=(('L','RB'),('M','RB'),('L','D'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_factory(count):
    rng=np.random.default_rng(0)
    groups=[np.arange(4),np.arange(4,6),np.arange(6,8)]
    mass=np.zeros((4000,8,10))
    for s in range(len(mass)):
        pp=rng.integers(0,10,10)
        for group in groups:
            oo=rng.choice(group,size=len(group),replace=True)
            weight=count[pp]/count[pp].sum()/(3*len(group))
            for o in oo:
                np.add.at(mass[s,o],pp,weight)
    def summary(v):
        avg=np.mean([v[g].mean(0) for g in groups],0)
        return dict(mean=np.average(avg,weights=count),interval=np.percentile(np.einsum('sop,op->s',mass,v),[10,90]).tolist(),people=avg.tolist())
    return summary


def main():
    old=ORIG.parent/'result.json'
    assert digest(old)=='bb6c7c625de9080c9f2ac1595cea814932eed10a0c0e0662bb5b1fa3a1a41a6f'
    assert (HERE/'replay.log').read_text().splitlines()[-1]=='OK'
    hist=json.loads(old.read_text())
    with np.load(ORIG/'runs/replay_data/bases.npz',allow_pickle=True) as z:
        people,cell,y=(z[k+'|all_loso'] for k in ('person','cell','y'))
    count=np.array([len(np.unique(cell[people==p])) for p in sc.PEOPLE])
    assert count.sum()==28
    summary=summary_factory(count)
    scores={}
    for oi,order in enumerate(sc.ORDERS):
        with np.load(HERE/f'count_{order}.npz') as filtered:
            for arm in ARMS:
                with np.load(HERE/f'{arm}_{order}.npz') as z:
                    for base in sc.BASES:
                        data={s:z[f'{base}|{s}'] for s in STAGES[1:5]}
                        data['count']=filtered[f'{arm}|{base}']
                        with np.load(ORIG/f'runs/s281_v66_cp/{arm}_{order}_{base}.npz') as cp:
                            data['CP']=cp['scores']
                        if arm!='control':
                            path=ORIG/f'runs/representation_streams/ttt18{order}_n395{arm.lower()}_base-all_loso-ensemble2.npz'
                            with np.load(path) as m:
                                assert np.array_equal(m['index'],np.arange(2655))
                                data['member']=m['proba']
                        for stage,value in data.items():
                            v=sc.person_accuracy(value,y,cell,people,sc.PEOPLE)
                            scores.setdefault((arm,base,stage),np.zeros((8,10)))[oi]=v
    result={}
    for key,bases in [('B2_pool',sc.BASES[:1]),('six_bases',sc.BASES)]:
        result[key]={}
        for arm in ARMS:
            for stage in STAGES[1:] if arm=='control' else STAGES:
                v=np.mean([scores[arm,b,stage] for b in bases],0)
                scores[arm,key+'@',stage]=v
                if stage in ('CP','member'):
                    r=hist['individual' if stage=='member' else key][arm]['official']
                    s=summary(v)
                    assert abs(s['mean']-r['mean'])<1e-9 and np.allclose(s['interval'],r['interval'],atol=1e-9)
        for a,b in PAIRS:
            d={s:scores[a,key+'@',s]-scores[b,key+'@',s] for s in STAGES}
            result[key][a+'-'+b]={s:summary(v) for s,v in d.items()}
            result[key][a+'-'+b].update({left+'->'+right:summary(d[right]-d[left]) for left,right in zip(STAGES,STAGES[1:])})
    (HERE/'attribution.json').write_text(json.dumps(result,indent=1))


if __name__=='__main__':
    main()
