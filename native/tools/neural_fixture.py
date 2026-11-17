"""One-time SOURCE-only golden capture; never imported by native execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    for key in ('helpers','package','data','out'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.helpers))
    from package_guard import instrument,frame
    from record import load_model,ft_clock_restart
    torch.set_num_threads(1);torch.manual_seed(3531701);torch.use_deterministic_algorithms(True)
    with np.load(args.data,allow_pickle=False) as data:
        rows=np.flatnonzero((data['subject']=='Dreyer2026Proteus/P01')&(data['session'].astype(str)=='1'))[:31]
        x=np.asarray(data['X'][rows],np.float32);channels=data['ch_names'].copy()
    if len(rows)!=31:raise ValueError('source P01/session1 must supply31 real windows')
    pins={'submission.py':'37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f','weights.npz':'7cf13696190983cd454880d6ecb8c049902a12ec694c1a24ff74cd4013a688a2'}
    for name,pin in pins.items():
        if hashlib.sha256((args.package/name).read_bytes()).hexdigest()!=pin:raise ValueError('frozen package identity differs:'+name)
    args.out.mkdir(exist_ok=False);result={'native_format':torch.tensor(1),'source_rows':torch.from_numpy(rows)};proof=[]
    for mode,count in [('known',8),('unknown',31)]:
        model=instrument(load_model(args.package,channels,'native353_neural_'+mode,'cpu'),mode=='unknown');ft_clock_restart(model)
        captures={key:[] for key in ('x','ids','probs','input','eye','rest','label','ft','ft_active','ft_stats','u','mix6','core1','contrast','identity','outer','runtime_scores')}
        inputs=sorted({ids[0][0] for _,ids in model.nets.groups}|set(model.ft.member.values()))
        result['input_ids']=torch.tensor(inputs,dtype=torch.int64)
        names=list(model.ft.member);original_refit=model.ft.refit;relab={};outer=model.routeB.stepx
        def refit(self,t,relab_fn=None):
            def capture(n):
                value=relab_fn(n)
                if value is not None:relab[n]=np.asarray(value).copy()
                return value
            return original_refit(t,capture if relab_fn is not None else None)
        def step_outer(self,window,u,mix,core1,extra,gc,gi):
            value=outer(window,u,mix,core1,extra,gc,gi)
            for key,v in [('u',u),('mix6',mix),('core1',core1),('outer',value)]:captures[key].append(np.asarray(v).copy())
            for key,v in [('contrast',gc),('identity',gi)]:captures[key].append(np.asarray(v if v is not None else [-1,0,0],np.float64))
            return value
        model.ft.refit=types.MethodType(refit,model.ft);model.routeB.stepx=types.MethodType(step_outer,model.routeB)
        begun=time.monotonic()
        for i in range(count):
            _,scores,_=frame(model,x[i]);trace=model.trace
            if model.dead or not trace.get('v6'):raise ValueError('reference neural path inactive/error')
            v=trace['v6'];ids=list(v['members']);tuned=v['p_ft']
            for key,value in [('x',x[i]),('runtime_scores',scores),('ids',np.asarray(ids,np.int64)),('probs',np.stack([v['members'][j] for j in ids])),
                              ('input',np.stack([model.nets.last_inputs[j] for j in inputs])),('eye',v['extra']),('rest',v['rest']),('label',v['label']),
                              ('ft',np.stack([tuned.get(s,np.zeros(3)) for s in names])),('ft_active',np.asarray([s in tuned for s in names],np.int64)),
                              ('ft_stats',np.asarray([model.ft.refits,model.ft.steps,model.ft.relabels,model.ft.capped,model.ft.over,model.ft.dtheta],np.float64))]:
                captures[key].append(np.asarray(value).copy())
            # Preserve the first real refit even if a later case reaches its time limit.
            if i in (0,20,30):
                partial={mode+'_'+key:torch.from_numpy(np.asarray(value)) for key,value in captures.items()}
                torch.save(dict(result,**partial),args.out/(mode+'_checkpoint.pt'))
                print(json.dumps(dict(checkpoint=mode,window=i,elapsed_seconds=time.monotonic()-begun,refits=model.ft.refits,steps=model.ft.steps)),flush=True)
        for key,value in captures.items():result[mode+'_'+key]=torch.from_numpy(np.asarray(value))
        for t,value in relab.items():result[mode+'_relabel_'+str(t)]=torch.from_numpy(value)
        if mode=='unknown' and (model.ft.refits!=2 or model.ft.steps!=40 or model.ft.dtheta<=0):raise ValueError('both real four-family fine-tuning refits must be active')
        proof.append(dict(mode=mode,windows=count,seconds=time.monotonic()-begun,refits=model.ft.refits,steps=model.ft.steps,relabels=model.ft.relabels,dtheta=model.ft.dtheta,errors=model.log6['err']))
        torch.save(result,args.out/'fixture.pt')
        (args.out/'proof.json').write_text(json.dumps(dict(cases=proof,accuracy_computed=False,task_labels_loaded=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
