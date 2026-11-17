"""Equal-cell proxy failure localization and paired people/order intervals."""
import numpy as np


def score(q, base, meta, people, y):
    persons = sorted(set(people))
    assert base.shape == (8,6,len(y))
    result = {}
    for subset,bidx in [('B2_pool',[0]),('six_bases',list(range(6)))]:
        values = np.zeros((len(persons),8,6))
        counts = []
        for pi,p in enumerate(persons):
            person = np.flatnonzero(people==p)
            cells = np.char.add(np.char.add(meta['stream'][person],'|'),meta['interface'][person])
            unique = np.unique(cells)
            counts.append(len(unique))
            for c in unique:
                for label in range(3):
                    r = person[(cells==c)&(y[person]==label)]
                    assert len(r) > 0
                    fail = q[r].argmax(1)!=y[r]
                    lowest = np.argsort(q[r].max(1),kind='stable')[:np.sum(fail)]
                    wrong = (base[:,bidx,:][:,:,r]!=y[r])
                    observed = (wrong*fail).mean((1,2))
                    shuffled = wrong.mean((1,2))*fail.mean()
                    confidence = wrong[:,:,lowest].sum((1,2))/(len(r)*len(bidx))
                    block = np.stack([observed,shuffled,confidence,
                                      np.full(8,np.mean(~fail)),1-wrong.mean((1,2)),
                                      np.full(8,fail.mean())],axis=1)
                    values[pi] += 100*block/(3*len(unique))
        keys = ['joint_wrong','independent_joint','confidence_joint','proxy_BA','actual_BA','proxy_error']
        counts = np.array(counts)
        def per(v):
            return (v[:,:4].mean(1)+v[:,4:6].mean(1)+v[:,6:].mean(1))/3
        result[subset] = {k:float(np.average(per(values[:,:,i]),weights=counts)) for i,k in enumerate(keys)}
        for i,key in ((1,'excess_vs_independence'),(2,'excess_vs_confidence')):
            d = values[:,:,0]-values[:,:,i]
            rng = np.random.default_rng(376)
            pp = rng.integers(len(persons),size=(4000,len(persons)))
            draws = np.concatenate([rng.integers(a,b,size=(*pp.shape,b-a)) for a,b in [(0,4),(4,6),(6,8)]],axis=2)
            v = d[pp[:,:,None],draws]
            res = (v[:,:,:4].mean(2)+v[:,:,4:6].mean(2)+v[:,:,6:].mean(2))/3
            ci = np.percentile(np.average(res,axis=1,weights=counts[pp]),[10,90])
            result[subset][key] = dict(official=float(np.average(per(d),weights=counts)),official80=ci.tolist(),
                people=float(per(d).mean()), people80=np.percentile(res.mean(1),[10,90]).tolist(),
                by_person=dict(zip(persons,per(d))))
    if abs(result['B2_pool']['actual_BA'] - 76.279335)>1e-4:
        raise ValueError('actual frozen B2_pool official endpoint changed')
    return result
