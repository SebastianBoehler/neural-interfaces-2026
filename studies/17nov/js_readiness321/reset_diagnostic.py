"""Measure original and variant self-repeat/reset/deepcopy without attributing failure or changing guards."""
import copy
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,'/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/2026-competition/tracks/bci_decoding')
import benchmark_utils.base_solver
from record import load_model
from reference import sha

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
PATHS={'oldJS':ROOT/'ni26_js321_20261117_orchestration4/prepared_jsall',
       'FIR':ROOT/'ni26_js321_20261117_shared2/prepared_jsall',
       'reuse':ROOT/'ni26_js321_20261117_inputreuse2/prepared_jsall'}


def tap(self,*a,**k):
    value=self.original_step(*a,**k);self.observed=np.asarray(value).copy();return value


def load(path,tag,channels,device):
    model=load_model(path,channels,tag,device);model.cp.original_step=model.cp.step;model.cp.step=types.MethodType(tap,model.cp)
    return model


def stream(model,x):
    result=[]
    for value in x:
        label=int(model.predict(torch.from_numpy(value[None]))[0]);t=model.trace
        result.append(dict(label=label,out=model.cp.observed.copy(),**{k:np.asarray(t[k]).copy() for k in ('core1','mix','u','shadow','out_v5')},
                           cc=np.asarray(t['js']['cc']).copy()))
    return result


def compare(a,b):
    report={};n=min(len(a),len(b))
    for key in ('out','core1','mix','u','shadow','out_v5','cc'):
        x=np.array([r[key] for r in a[:n]]);y=np.array([r[key] for r in b[:n]])
        report[key]=dict(exact=bool(np.array_equal(x,y)),max_abs=float(abs(x-y).max()),max_relative=float((abs(x-y)/np.maximum(abs(x),1e-12)).max()))
    report['label_changes']=sum(x['label']!=y['label'] for x,y in zip(a[:n],b[:n]));return report


def metadata(model):
    values={'core1_W':model.core1.W,'core2_W':model.core2.W,'view0':model.view_V[0],'view1':model.view_V[1],'weights':model.w}
    arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),strides=list(v.strides),sha256=hashlib.sha256(v.tobytes(order='A')).hexdigest()) for k,v in values.items()}
    states=[b for _,members in model.js_bank.groups for _,net in members for name,b in net.m.named_buffers() if name.rsplit('.',1)[-1] in ('s1','s2','n')]
    return dict(arrays=arrays,aux_stream_buffers=len(states),aux_all_zero=all(float(b.abs().sum())==0 for b in states),
                aux_input_n=model.js_bank.groups[0][0].n,core_n=model.core1.n,ft_refits=model.ft.refits,cp_refits=model.cp.refits)


def main():
    start=time.monotonic();torch.set_num_threads(1)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit A100 required')
    with np.load(ROOT/'ni26_s304_coral_prod02_20261116/data/proteus_windows.npz',allow_pickle=False) as z:x=z['X'][:45].copy();channels=z['ch_names']
    report=dict(status='reset_selfrepeat_diagnostic_only',accuracy_computed=False,original_guard_failures_preserved=True,cases={},
                torch=torch.__version__,dtype=str(x.dtype),threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
                blas={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    for device in ('cuda','cpu'):
        for name,path in PATHS.items():
            n=45 if device=='cuda' else 17;model=load(path,name+device,channels,device);initial=metadata(model)
            cold=stream(model,x[:n]);model.reset_state();reset_meta=metadata(model);reset=stream(model,x[:17])
            model.reset_state();left,right=copy.deepcopy(model),copy.deepcopy(model);copy_meta=metadata(left);copied=[]
            for i in range(17):
                copied+=stream(left,x[i:i+1])
                if i<5:stream(right,x[-1-i:-i or None])
            fresh=load(path,'fresh_'+name+device,channels,device);again=stream(fresh,x[:17])
            report['cases'][name+'_'+device]=dict(cold_reset=compare(cold,reset),cold_deepcopy_interleaved=compare(cold,copied),
                     cold_fresh_selfrepeat=compare(cold,again),initial=initial,reset=reset_meta,deepcopy=copy_meta,
                     bundle_sha256=sha(path/'submission.py'),weights_sha256=sha(path/'weights.npz'))
            del model,left,right,fresh;torch.cuda.empty_cache()
    report['seconds']=time.monotonic()-start;Path('reset_diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{c:v[c]['out'] for c in ('cold_reset','cold_deepcopy_interleaved','cold_fresh_selfrepeat')} for k,v in report['cases'].items()}),flush=True)


if __name__=='__main__':main()
