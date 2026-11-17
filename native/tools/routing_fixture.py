"""Capture actual SOURCE-window stage inputs/outputs; never load task labels or score."""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--helpers',type=Path,required=True);parser.add_argument('--package',type=Path,required=True);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.helpers))
    from package_guard import instrument,frame
    from record import load_model,ft_clock_restart
    torch.set_num_threads(1);torch.manual_seed(3531701);torch.use_deterministic_algorithms(True)
    with np.load(args.data,allow_pickle=True) as z:
        rows=np.flatnonzero((z['subject']=='Dreyer2026Proteus/P01')&(z['session'].astype(str)=='1'))
        X=z['X'][rows].copy();channels=z['ch_names'].copy()
    if len(X)<80:raise ValueError('actual P01 source session insufficient')
    args.out.mkdir();result={'native_format':torch.tensor(1)};proof=[]
    for mode,count in [('known',80),('unknown',215)]:
        model=instrument(load_model(args.package,channels,'native353_reference_'+mode,'cpu'),mode=='unknown');ft_clock_restart(model)
        captures={k:[] for k in ('x','u','mix','core1','shadow','route','contrast','identity','p','count','cp','recognized')}
        route_step=model.route.step;count_step=model.cf.step;cp_step=model.cp._original_step
        def routing(self,x,u,mix,core1,shadow):
            out=route_step(x,u,mix,core1,shadow)
            for key,value in [('x',x),('u',u),('mix',mix),('core1',core1),('shadow',shadow),('route',out)]:captures[key].append(np.asarray(value).copy())
            for key in ('contrast','identity'):
                gate=self.last[key];captures[key].append(np.asarray(gate if gate is not None else [-1,0,0],dtype=np.float64))
            return out
        def counted(self,x,p):
            out=count_step(x,p);captures['p'].append(np.asarray(p).copy());captures['count'].append(np.asarray(out).copy());return out
        def personalized(self,x,p,o,recognized):
            out=cp_step(x,p,o,recognized);captures['cp'].append(np.asarray(out).copy());captures['recognized'].append(int(recognized));return out
        model.route.step=types.MethodType(routing,model.route);model.cf.step=types.MethodType(counted,model.cf);model.cp._original_step=types.MethodType(personalized,model.cp)
        for i in range(count):frame(model,X[i%len(X)])
        for key,value in captures.items():result[mode+'_'+key]=torch.from_numpy(np.asarray(value))
        proof.append(dict(mode=mode,windows=count,source_rows=rows.tolist(),rec=model.log6['rec'],cp_refits=model.cp.refits,cp_fused=model.cp.fused,cp_errors=model.cp.log['err'],count_errors=model.cf.log['err']))
        if model.cp.log['err'] or model.cf.log['err'] or (mode=='unknown' and (model.cp.refits!=len(range(model.cp.t0,count,model.cp.k)) or model.cp.fused!=125)):raise ValueError('late stage reference inactive/error')
    torch.save(result,args.out/'fixture.pt');json.dump(dict(cases=proof,accuracy_computed=False,package_sha256=hashlib.sha256((args.package/'submission.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),open(args.out/'proof.json','w'),indent=2)


if __name__=='__main__':main()
