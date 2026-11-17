"""Same-weight optimized raw/active/off/reset/late-CP parity before fresh worker probes."""
import copy
import json
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
from shared_raw import verify_raw

OLD=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_orchestration4/prepared_jsall')
BASE=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_active5/draft_base')
DATA=Path('/mnt/lustre/work/gehler/gfh527/ni26_s304_coral_prod02_20261116/data/proteus_windows.npz')


def capture(self,*a,**k):
    value=self.original_step(*a,**k);self.observed=np.asarray(value).copy();return value


def load(path,tag,channels,unrecognized=False):
    model=load_model(path,channels,tag,'cuda')
    if Path(type(model).step.__globals__['__file__']).resolve()!=(path/'submission.py').resolve():raise ValueError('module path differs')
    model.cp.original_step=model.cp.step;model.cp.step=types.MethodType(capture,model.cp)
    if unrecognized:model.route.contrast.step=lambda *a:None;model.route.identity.step=lambda *a:None
    return model


def frame(model,x):
    label=int(model.predict(torch.from_numpy(x[None].astype(np.float32)))[0])
    return label,model.cp.observed.copy(),copy.deepcopy(model.trace)


def main():
    start=time.monotonic();ws=Path.cwd();torch.set_num_threads(1)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit A100 source guard required')
    complete=json.loads((ws/'complete.json').read_text())
    for name,digest in complete['files'].items():
        if sha(ws/'prepared_jsall'/name)!=digest:raise ValueError('optimized artifact changed')
        if name!='submission.py' and digest!=sha(OLD/name):raise ValueError('a model/weight file changed')
    with np.load(DATA,allow_pickle=False) as z:x=z['X'][:215].copy();channels=z['ch_names']
    probe=load(ws/'prepared_jsall','raw_shared321',channels);g=type(probe).step.__globals__
    raw=verify_raw(g,'cuda');del probe;torch.cuda.empty_cache();cases={}
    for mode,n in (('known',45),('unrecognized',215)):
        old=load(OLD,'old_shared321_'+mode,channels,mode=='unrecognized')
        new=load(ws/'prepared_jsall','new_shared321_'+mode,channels,mode=='unrecognized')
        original=load(BASE,'original_shared321_'+mode,channels,mode=='unrecognized')
        off=load(ws/'draft_off','off_shared321_'+mode,channels,mode=='unrecognized')
        maximum=0.;reference=[]
        for t,value in enumerate(x[:n]):
            a,pa,ta=frame(old,value);b,pb,tb=frame(new,value)
            maximum=max(maximum,float(abs(pa-pb).max()));reference.append(pb)
            if a!=b or maximum>1e-8:raise ValueError('same-weight optimized final output parity failed')
            if getattr(new,'js_reuse',False):
                if new.js_step!=t+1 or new.js_done!=t+1 or new.js_bank.groups[0][0].n!=0:
                    raise ValueError('current-step/once-only/unused auxiliary EA guard failed')
                if not np.array_equal(new.nets.last_inputs[35],old.nets.last_inputs[35]):
                    raise ValueError('original current input was changed or mismatched')
            for field in ('mix','shadow','core1','u','out_v5'):
                if not np.array_equal(ta[field],tb[field]):raise ValueError('original inner path changed: '+field)
            for field in ('contrast','identity'):
                va,vb=ta[field],tb[field]
                if not (va is None and vb is None) and not np.array_equal(va,vb):raise ValueError('gate changed')
            if not np.array_equal(old.ft.lab,new.ft.lab) or not np.array_equal(old.ft.wt,new.ft.wt):raise ValueError('original FTlabel chain changed')
            if t<45:
                c,pc,_=frame(original,value);d,pd,_=frame(off,value)
                if c!=d or not np.array_equal(pc,pd):raise ValueError('new-source disabled final identity failed')
        classes=[type(net).__name__ for _,members in new.js_bank.groups for _,net in members]
        if classes!=['SharedFIRSJit']*6 or new.js_calls!=n or new.log6['err'] or old.log6['err'] or new.cp.log['err']:
            raise ValueError('shared FIR activity/runtime failed')
        if mode=='unrecognized' and (new.recB or new.ft.refits<=0 or new.cp.refits<=0 or new.cp.fused<=0):raise ValueError('late FT/CP path inactive')
        new.reset_state();left,right=copy.deepcopy(new),copy.deepcopy(new)
        for t in range(17):
            _,p,_=frame(left,x[t])
            if not np.array_equal(p,reference[t]):raise ValueError('reset/prefix/deepcopy parity failed')
            if t<5:frame(right,x[-1-t])
        cases[mode]=dict(input_reuse_checked=bool(getattr(new,'js_reuse',False)),windows=n,max_output_error=maximum,inner_bitwise=True,off45_bitwise=True,ft_labels_bitwise=True,
                         reset_deepcopy_prefix=True,shared_classes=classes,cc_calls=n,recognized_at=old.log6['rec'],
                         old_ft_refits=old.ft.refits,new_ft_refits=new.ft.refits,old_cp_refits=old.cp.refits,new_cp_refits=new.cp.refits,
                         old_cp_fused=old.cp.fused,new_cp_fused=new.cp.fused)
    report=dict(status='same_weight_sharedFIR_raw_active_reset_guards_passed',source_sha256=sha(ws/'source.sha256'),
                complete_sha256=sha(ws/'complete.json'),parent_complete_sha256=complete['parent_complete_sha256'],raw=raw,cases=cases,
                execution_variant=complete['execution_variant'],seconds=time.monotonic()-start,accuracy_computed=False,scope='engineering real source prefix;215 concatenated length stress;not held/returning score or workerbar recovery')
    (ws/'optimization_guard.json').write_text(json.dumps(report,indent=2));print(report['status'],flush=True)


if __name__=='__main__':main()
