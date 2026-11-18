"""Independent single-window policy implementation compared with the actual vector actor."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import softmax

ACTOR=Path('/mnt/lustre/work/gehler/gfh527/ni26_policy395_actor_v1_20261118')
sys.path.insert(0,str(ACTOR))
from substrate import ORDERS,PEOPLE,metadata,orders,outer,sha,source_bank

B=np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0.,-2/np.sqrt(6)]])
FIX=[(np.eye(2)*np.arctanh(.5)).tolist(),[np.log(np.sqrt(20/3))]*2,0.]


def predict(current,other,params,flat):
    gain,diag,cross=params
    diag=np.exp(diag)
    L=np.array([[diag[0],0.],[cross,diag[1]]])
    H=L@L.T+1e-6*np.eye(2)
    total=np.zeros(2)
    out=[]
    for i,p in enumerate(current):
        corr=1.5*np.tanh((B@((2*np.tanh(gain))@np.linalg.solve(H,total)))/1.5)
        if i==0:
            assert np.array_equal(corr,np.zeros(3))
        out.append(softmax(np.log(p)-corr))
        q=other[i%len(other)]
        total+=B.T@q
        H+=np.eye(2)/3 if flat else B.T@(np.diag(q)-np.outer(q,q))@B
    return np.stack(out)


def main():
    assert sha(ACTOR/'source.sha256')=='3df961858b0320b58962fc07fa9e05e7bb0089e057158bb72e5a8b8009727c45'
    meta=metadata();perms=orders()
    maximum=first=0.;count=0
    for p in PEOPLE:
        rows,base=outer(p,meta)
        rec=json.loads((ACTOR/(p+'.json')).read_text())
        assert rec['outputs']==sha(ACTOR/(p+'.npz'))
        with np.load(ACTOR/(p+'.npz')) as z:
            for seed in (0,1):
                banks,_=source_bank(p,seed,meta,perms)
                proof=rec['proofs'][str(seed)]
                for o,index in perms.items():
                    donor=next(v[4] for v in banks if v[1]==o)
                    for s in np.unique(meta['stream'][rows]):
                        rr=index[meta['stream'][index]==s]
                        pos=np.searchsorted(rows,rr)
                        native=base[seed]['native|'+o][pos]
                        for arm,pars,other in [('learned',proof['learned']['params'],native),('mean',proof['mean']['params'],native),
                                               ('donor',proof['learned']['params'],donor),('fixed',FIX,native)]:
                            direct=predict(native,other,pars,arm=='mean')
                            saved=z[f'{seed}|{arm}|{o}'][pos]
                            assert np.array_equal(saved.argmax(1),direct.argmax(1))
                            assert saved[0].argmax()==native[0].argmax()
                            first=max(first,float(abs(saved[0]-native[0]).max()))
                            maximum=max(maximum,float(abs(saved-direct).max()))
                            count+=len(rr)
    assert maximum<1e-9
    Path('result.json').write_text(json.dumps(dict(posterior_max=maximum,first_row_max=first,
              single_window_predictions=count,labels_opened=False,source=sha('source.sha256')),indent=2))


if __name__=='__main__':
    main()
