import json
from common import *

BASE=['C4','N19','N19E','N19P']+['R'+str(i) for i in range(8)]


def summarize(p,c,g,y,bank):
    names=list(bank)
    f=np.column_stack(list(bank.values()))
    assert np.isfinite(f).all()
    ans=[]
    for pp in sorted(set(p)):
        cells=[]
        for cc in sorted(set(c[p==pp])):
            runs=[]
            for gg in sorted(set(g[c==cc])):
                mm=g==gg
                yr,v=y[mm],f[mm]
                runs.append(np.stack([auc(yr,v),100*(v[yr==2]>=0).mean(0),100*(v[yr==1]<0).mean(0)]))
            cells.append(np.mean(runs,0))
        ans.append(np.mean(cells,0))
    arr=np.array(ans)
    result={n:arr[:,:,i] for i,n in enumerate(names)}
    if 'R0' in result:result['random4']=np.mean([result['R'+str(i)] for i in range(8)],0)
    return result


def stats(vals,rng):
    replicates=rng.integers(len(vals),size=(10000,len(vals)))
    return dict(mean=vals.mean(0).tolist(),ci=np.quantile(vals[replicates].mean(1),[.1,.9],axis=0).tolist())


def main():
    rng=np.random.default_rng(397);out={}
    ref=json.loads((Path('../ni26_semantics393_v1_20261118/result.json')).read_text())
    for k in DATA:
        other='P' if k=='S' else 'S'
        with np.load(DATA[k]) as z:p,c,g=fields(k,z)
        roster=sorted(set(p))
        banks={a:np.full(len(p),np.nan) for a in BASE}
        df={}
        for i,pid in enumerate(roster):
            with np.load(f'{k}_{i}.npz') as z:
                r=np.flatnonzero(p==pid)
                assert np.array_equal(z['index'],r) and np.array_equal(z['train'],np.flatnonzero(p!=pid))
                for a in BASE:banks[a][r]=z[a]
                df[pid]=float(z['N19_df'])
        with np.load(f'{other}_-1.npz') as z:
            assert np.array_equal(z['index'],np.arange(len(p)))
            cross={a:z[a] for a in BASE}
        with np.load('shin'+k+'.npz') as z:
            shin={a:z[a] for a in z.files if not a.startswith('source')}
            src={a:float(z[a]) for a in z.files if a.startswith('source')}
        with np.load(DATA[k]) as z:y=z['y']
        out[k]={'source_shin_auc':src,'wide_df':df}
        for name,matrix in [('direct_within',banks),('direct_cross',cross),('shin',shin)]:
            res=summarize(p,c,g,y,matrix)
            if name=='shin':
                old=ref['target']['Scherer' if k=='S' else 'PROTEUS']['semantic']
                assert abs(res['raw'][:,0].mean()-old)<1e-8
                contrasts=[('wide','C4'),('Aexpand','C4'),('Wexpand','C4'),('Wexpand','Aexpand'),
                           ('Wwide','W4'),('minusAwide','minusA4')]
            else:contrasts=[('N19E','C4'),('C4','random4'),('N19E','N19P'),('N19','N19E')]
            out[k][name]={'arms':{a:stats(v,rng) for a,v in res.items()},
                          'contrasts':{a+'-'+b:stats(res[a]-res[b],rng) for a,b in contrasts}}
            if name=='shin':
                # Only direct source-midpoint discriminants carry an identified class threshold.
                for grp in out[k][name].values():
                    for val in grp.values():
                        val['mean']=val['mean'][:1]
                        val['ci']=[q[:1] for q in val['ci']]
    Path('result.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
