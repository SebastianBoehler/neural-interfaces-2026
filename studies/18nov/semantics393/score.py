import csv
import gzip
import json
from pathlib import Path

import numpy as np
from features import ROOT
from produce import digest


def main():
    pre=json.loads(Path('before.json').read_text())
    for name,h in pre['hashes'].items():assert digest(name)==h
    rng=np.random.default_rng(393)
    output={'source':{},'target':{}}
    for kind,group in pre['source_aucs'].items():
        vals=100*np.array(list(group.values()))
        draws=rng.choice(vals,size=(10000,len(vals))).mean(1)
        output['source'][kind]=dict(mean=float(vals.mean()),ci=np.quantile(draws,[.1,.9]).tolist())
    for corpus,path in [('Scherer',ROOT/'data/scherer2015/windows.npz'),('PROTEUS',ROOT/'data/proteus_windows.npz')]:
        with np.load(path) as z:
            p,s,interface,r,y=(z[k] for k in ('subject','session','interface','run','y'))
        if corpus=='Scherer':
            with gzip.open(ROOT/'runs/diag/external_semantics351/scherer_rows.tsv.gz','rt') as f:
                rows=list(csv.DictReader(f,delimiter='\t'))
            assert len(rows)==len(p) and all(int(v['row'])==i and p[i].split('/')[-1]==v['person'] and
                                            s[i]==v['session'] for i,v in enumerate(rows))
            r=np.array([v['run'] for v in rows])
        bank=np.load(corpus+'.npy')
        assert bank.shape==(len(y),132)
        people=[]; run_count=0; empty_count=0; reviewed=0
        for person in sorted(set(p)):
            cells=[]
            for day,interface_name in sorted(set(zip(s[p==person],interface[p==person]))):
                mask=(p==person)&(s==day)&(interface==interface_name)
                runs=[]
                for run in sorted(set(r[mask])):
                    a=bank[mask&(r==run)&(y==2)]
                    b=bank[mask&(r==run)&(y==1)]
                    if not len(a) or not len(b):
                        empty_count+=1
                        continue
                    diff=a[:,None,:]-b[None,:,:]
                    runs.append(100*((diff>0)+.5*(diff==0)).mean((0,1)))
                    run_count+=1
                    reviewed+=len(a)+len(b)
                assert runs
                cells.append(np.mean(runs,axis=0))
            people.append(np.mean(cells,axis=0))
        v=np.stack(people);m=v.mean(0)
        boot=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
        assert reviewed==(1420 if corpus=='Scherer' else np.isin(y,(1,2)).sum())
        if corpus=='Scherer':assert run_count==142 and empty_count==0
        output['target'][corpus]=dict(people=len(v),runs=run_count,excluded_runs=empty_count,rows=int(reviewed),
            semantic=m[0],generic=m[1],word=m[2],minus_arithmetic=m[3],ci=np.quantile(boot[:,0],[.1,.9]).tolist(),
            semantic_minus_generic=m[0]-m[1],contrast_ci=np.quantile(boot[:,0]-boot[:,1],[.1,.9]).tolist(),
            null_mean=float(m[4:].mean()),null90=np.quantile(m[4:],.9),per_person=v[:,0].tolist(),
            above50=int((v[:,0]>50).sum()))
    Path('result.json').write_text(json.dumps(output,indent=2)+'\n')
    print(output)
if __name__=='__main__':main()
