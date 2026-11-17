"""Trace-only real-activation numeric failure; never releases/scorers/labels."""
import copy
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_moments332_v1_20261117')
sys.path.insert(0,str(ROOT))
import source332 as src
from views332 import prepare,MomentBN,direct


def main():
    torch.set_num_threads(6);d,fit,test,donor,perms,nets,original,meta=src.load('P10',0,'cpu')
    rows=test[d['stream'][test]==d['stream'][test[0]]][:12];results={}
    for label,(model,scale) in nets.items():
        x=torch.from_numpy(d['X'][rows]/scale);views,_=prepare(model,x);captured=[]
        layer=next(m for m in views['B'].body.modules() if isinstance(m,src.session_bn.SessionBN))
        hook=layer.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
        src.modes.probabilities(views['B'],d['X'][rows]/scale,'cpu');hook.remove();activation=captured[0]
        results[label]={'input_shape':list(activation.shape),'input_max':float(activation.abs().max()),
            'running_mean_max':float(layer.running_mean.abs().max()),'running_var_min':float(layer.running_var.min()),'eps':layer.eps,'errors':{}}
        for mode in ('M','V','B','D'):
            custom=copy.deepcopy(layer);custom.__class__=MomentBN;custom.mode332=mode
            value=activation if mode!='D' else torch.cat([activation,activation.flip(0)])
            with torch.no_grad():
                reference=direct(copy.deepcopy(custom).double(),value.double(),mode)
                for dtype in (torch.float64,torch.float32):
                    adapted=copy.deepcopy(custom).to(dtype);z=value.to(dtype)
                    a=adapted(z);b=direct(adapted,z,mode)
                    results[label]['errors'][mode+'|'+str(dtype)]={'direct_same_dtype_error':float((a-b).abs().max()),
                        'cumsum_vs_f64_error':float((a.double()-reference).abs().max()),'direct_vs_f64_error':float((b.double()-reference).abs().max())}
    path=Path(__file__).resolve().parent/'results.json'
    with path.open('x') as out:json.dump(results,out,indent=2)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
