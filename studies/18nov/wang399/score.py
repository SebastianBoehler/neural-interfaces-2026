import json
from pathlib import Path
import sys
import hashlib
import numpy as np
from common import DATA,fields,auc


def main(reserve):
    rng=np.random.default_rng(399)
    out={}
    for k in DATA:
        proof=json.loads(Path('proof'+k+'.json').read_text())
        path=Path('scores'+k+'.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==proof['bank_sha256']
        with np.load(path) as z:
            names=[a for a in z.files if a.endswith('null') or ('session' in a)==reserve]
            pred={a:z[a] for a in names}
        pred=np.column_stack(list(pred.values()))
        with np.load(DATA[k]) as z:p,c,g=fields(k,z);y=z['y']
        people=[]
        for pp in sorted(set(p)):
            cells=[]
            for cc in sorted(set(c[p==pp])):
                ranks=[]
                for gg in sorted(set(g[c==cc])):
                    mask=g==gg
                    ranks.append(auc(y[mask],pred[mask]))
                cells.append(np.mean(ranks,axis=0))
            people.append(np.mean(cells,axis=0))
        people=np.stack(people); out[k]={}
        draws=rng.integers(len(people),size=(10000,len(people)))
        start=0;arms={}
        for name in names:
            length=64 if name.endswith('null') else 1
            v=people[:,start:start+length];start+=length
            if length==64:
                means=v.mean(0)
                out[k][name]=dict(mean=means.mean(),p90=np.quantile(means,.9),
                                  at_least_true=int((means>=means[-1]-1e-10).sum()),total=64,true=means[-1])
            else:
                arms[name]=v[:,0]
        keys=(['session1','session2','session3'] if reserve else ['arith','foil','drift','order'])
        for tag in ('4','wide'):
            if reserve:arms[tag+'first_minus_last']=arms[tag+'session1']-arms[tag+'session3']
            else:
                assert abs(np.mean(arms[tag+'arith'])-out[k][tag+'null']['true'])<1e-8
                for q in keys[1:]:arms[tag+'arith-'+q]=arms[tag+'arith']-arms[tag+q]
                out[k][tag+'source']=list(np.mean(proof[tag]['source_auc'],axis=1))
        if not reserve:arms['wide-4']=arms['widearith']-arms['4arith']
        for name,v in arms.items():
            out[k][name]=dict(mean=v.mean(),ci=np.quantile(v[draws].mean(1),[.1,.9]).tolist())
    Path('reserve.json' if reserve else 'result.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1]=='reserve')
