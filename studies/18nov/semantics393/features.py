from itertools import combinations
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
CH=('Cz','P3','P4','Pz')
PAIRS=np.array(list(combinations(range(4),2)))
BANDS=[(2,4),(4,8),(8,13),(13,20),(20,30),(30,40)]


def extract(x):
    v=x[:,PAIRS[:,0]].astype(np.float64)-x[:,PAIRS[:,1]]
    v-=v.mean(-1,keepdims=True)
    ps=np.abs(np.fft.rfft(v*np.hanning(480),axis=-1))**2
    hz=np.fft.rfftfreq(480,1/120)
    powers=np.stack([ps[:,:,(hz>=a)&(hz<b)].sum(-1) for a,b in BANDS],axis=-1)
    result=np.log((powers+1e-12)/(powers.sum(-1,keepdims=True)+6e-12))
    assert np.isfinite(result).all()
    return result.reshape(len(x),-1)


def load(path,source=False):
    with np.load(path) as z:
        names=z['ch_names'].tolist()
        native=set(names)-set(z['interpolated']) if 'interpolated' in z.files else set(names)
        assert set(CH)<=native
        x=z['X'][:,[names.index(c) for c in CH]]
        d={k:z[k] for k in ('subject','session','run','onset','interface')}
        if source:
            d.update(y=z['y'],classes=z['classes'])
            if 'win' in z.files:d['win']=z['win']
    f=np.concatenate([extract(x[i:i+64]) for i in range(0,len(x),64)])
    assert np.allclose(extract(x[[4,1,0]]),f[[4,1,0]],rtol=1e-12,atol=1e-12)
    d['f']=f
    return d,native


def data393(kind,paths):
    pairs,rows,natives={},[],[]
    for path in paths:
        d,chs=load(path,True)
        natives.append(chs)
        expect=['rest','subtraction'] if kind=='A' else ['baseline','WG']
        assert list(d['classes'])==expect
        for p in sorted(set(d['subject'])):
            assert p not in pairs
            pairs[p]=[]
            ix=np.flatnonzero(d['subject']==p)
            groups=np.char.add(np.char.add(d['session'][ix],'|'),d['run'][ix].astype(str))
            assert len(set(groups))==3
            for g in sorted(set(groups)):
                r=ix[groups==g]
                if kind=='W':
                    assert len(r)==40 and np.array_equal(d['win'][r],np.tile([0,1],20))
                    q,q2=r[::2],r[1::2]
                else:
                    q,q2=np.split(r,2)
                assert len(q)==20 and np.allclose(d['onset'][q2]-d['onset'][q],4)
                assert np.array_equal(d['y'][q2],d['y'][q])
                q=q[np.argsort(d['onset'][q])]
                pos=q[d['y'][q]==1];neg=q[d['y'][q]==0]
                assert len(pos)==len(neg)==10
                pairs[p].append(d['f'][pos]-d['f'][neg])
                rows.append((p,d['y'][q],d['f'][q]))
    assert len(set(map(frozenset,natives)))==1
    return pairs,rows,natives[0]
