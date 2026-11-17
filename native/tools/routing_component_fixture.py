"""Real source-component fixture, not a whole-model reference or accuracy study."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    sys.path.insert(0,str(args.project/'2026-competition/tracks/bci_decoding'));torch.set_num_threads(1);limits=threadpool_limits(1)
    spec=importlib.util.spec_from_file_location('frozen353_source_components',args.package/'submission.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with np.load(args.package/'weights.npz') as z:weights={k:z[k] for k in z.files}
    with np.load(args.project/'data/proteus_windows.npz',allow_pickle=True) as z:
        rows=np.flatnonzero((z['subject']=='Dreyer2026Proteus/P01')&(z['session'].astype(str)=='1'));X=z['X'][rows].copy()
    args.out.mkdir();values={'native_format':torch.tensor(1)};records=[]
    for mode,count in [('known',80),('unknown',215)]:
        c1=m.TangentCore(m.with_prefix(weights,'ts_'));c2=m.TangentCore(m.with_prefix(weights,'ts2_'));tail=m.Tail(weights);route=m.Route(weights,float(weights['member_weights'][0]/weights['member_weights'].sum()));cf=m.CountFilter(weights);cp=m.CausalPers(weights)
        if mode=='unknown':route.contrast.step=lambda *a:None;route.identity.step=lambda *a:None
        result={k:[] for k in ('x','u','mix','core1','shadow','route','contrast','identity','p','count','cp','recognized')};recognized=False
        for i in range(count):
            x=X[i%len(X)];p,f=c1.step(x);q=c2.step(x)[0];d=f-weights['tsem_m'];u=np.r_[d@weights['tsem_V'],d@weights['route_V_hi']];mix=.5*(p+q)
            shadow=tail.step(x,u[:route.n_tsem],mix);out=route.step(x,u,mix,p,shadow);gc,gi=route.last['contrast'],route.last['identity'];recognized=recognized or any(g is not None and g[2]>0 for g in (gc,gi));counted=cf.step(x,out);personal=cp.step(x,out,counted,recognized)
            for key,value in dict(x=x,u=u,mix=mix,core1=p,shadow=shadow,route=out,p=out,count=counted,cp=personal,recognized=int(recognized),contrast=gc if gc is not None else [-1,0,0],identity=gi if gi is not None else [-1,0,0]).items():result[key].append(np.asarray(value).copy())
            if i%20==0:print(mode,i,flush=True)
        if cp.log['err'] or cf.log['err'] or (mode=='unknown' and (cp.refits!=len(range(cp.t0,count,cp.k)) or cp.fused!=125)):raise ValueError('actual component lateCP failed')
        for key,value in result.items():values[mode+'_'+key]=torch.from_numpy(np.asarray(value))
        records.append(dict(mode=mode,windows=count,cp_refits=cp.refits,cp_fused=cp.fused,cp_errors=cp.log['err'],cf_errors=cf.log['err']))
        # Preserve each complete case independently; incomplete unknown is never called complete.
        torch.save(values,args.out/(mode+'_complete.pt'));json.dump(dict(cases=records,scope='SOURCE_COMPONENTS_ONLY_actualcore1/core2halfmixture_notfullensemble',accuracy_computed=False),open(args.out/'progress.json','w'),indent=2)
    torch.save(values,args.out/'fixture.pt');json.dump(dict(cases=records,status='SOURCE_COMPONENT_FIXTURE_COMPLETE_NOT_WHOLE_MODEL',source_rows=rows.tolist(),accuracy_computed=False),open(args.out/'proof.json','w'),indent=2)


if __name__=='__main__':main()
