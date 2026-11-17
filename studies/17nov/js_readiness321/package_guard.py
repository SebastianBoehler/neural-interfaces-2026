"""Actual generated-bundle identity and JS-all component checks on real EEG, without accuracy scoring."""
import ast
import copy
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from reference import sha

OFFICIAL=Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/2026-competition/tracks/bci_decoding')
sys.path.insert(0,str(OFFICIAL))
import benchmark_utils.base_solver  # Real owning base; prevent record helper's optional stub.


def load(path,tag,channels):
    spec=importlib.util.spec_from_file_location('owning_record_helper',Path.cwd()/'record.py')
    record=importlib.util.module_from_spec(spec);spec.loader.exec_module(record)
    model=record.load_model(path,channels,tag,'cpu')
    expected=(path/'submission.py').resolve()
    if Path(type(model).step.__globals__['__file__']).resolve()!=expected:
        raise ValueError('loaded module path differs from requested generated bundle')
    return model


def capture_cp(self,*args,**kwargs):
    out=self._original_step(*args,**kwargs);self._observed=np.asarray(out).copy();return out


def instrument(model,unrecognized=False):
    model.cp._original_step=model.cp.step;model.cp.step=types.MethodType(capture_cp,model.cp)
    if unrecognized:
        model.route.contrast.step=lambda *args:None;model.route.identity.step=lambda *args:None
    return model


def frame(model,x):
    label=int(model.predict(torch.from_numpy(x[None].astype(np.float32)))[0])
    return label,model.cp._observed.copy(),copy.deepcopy(model.trace)


def owning_mix():
    text=(Path.cwd()/'alib.py').read_text();tree=ast.parse(text)
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mix')
    a=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FAM' for t in n.targets))
    ns={'np':np,'FAM':ast.literal_eval(a.value)}
    exec(compile(ast.Module(body=[f],type_ignores=[]),'owning_alib_mix','exec'),ns)
    return ns


def reference_mix(model,trace,z,reference):
    members=trace['v6']['members'];ft=trace['v6']['p_ft'];w=model.w
    def mean(ids):
        weights=w[1+np.asarray(ids)];return (weights[:,None]*np.stack([members[i] for i in ids])).sum(0)/weights.sum()
    def group(slot):
        p=mean(model.ft.slots[slot])
        if slot in ft:
            p=np.clip(p+ft[slot]-members[model.ft.member[slot]],1e-6,None);p/=p.sum()
        return p
    shallow=[i for i in members if f'net{i}_time_w' in z]
    tf=[i for i in members if f'net{i}_fbc_n_bands' in z]
    teacher=[i for i in members if f'net{i}_jit_file' in z]
    if len(shallow)!=10 or len(tf)!=4 or teacher!=[34]:raise ValueError('original family classification changed')
    partition=lambda ids,aligned:[i for i in ids if bool(z[f'net{i}_align'])==aligned]
    if [len(partition(ids,a)) for ids in (shallow,tf) for a in (True,False)]!=[5,5,2,2]:
        raise ValueError('original EA/raw family weights changed')
    residual=trace['mix']-w[0]*trace['core1']-sum(w[1+i]*p for i,p in members.items())
    mem={'c1':trace['core1'],'hb':residual/w[-1],1:mean(partition(tf,True)),2:mean(partition(tf,False)),
         3:mean(partition(shallow,True)),4:mean(partition(shallow,False)),9:mean(list(range(41,47))),10:members[34],
         'sc_fbraw':group('fbraw'),'sc_sinc':group('sinc'),'sc_fbl':group('fbl'),
         'nm_fb_cc30':trace['js']['cc'],'sc_rest':trace['v6']['rest']}
    reference['members']=lambda bn,p:mem
    return reference['mix']('B2_pool','real-package-frame',['core1','hilb','tf','shallow','teacher'],
                            [(9,1.),('sc_fbraw',1.),('sc_sinc',1.),('sc_fbl',2.),('nm_fb_cc30',1.),('sc_rest',1.)])


def same_gate(a,b):
    return a is None and b is None or a is not None and b is not None and np.array_equal(a,b)


def main():
    start=time.monotonic();ws=Path.cwd();torch.set_num_threads(1)
    with np.load(ws/'data/proteus_windows.npz',allow_pickle=False) as z:
        x=z['X'].copy();subject=z['subject'];session=z['session'];channels=z['ch_names']
    rows=np.flatnonzero((subject=='Dreyer2026Proteus/P01')&(session==session[subject=='Dreyer2026Proteus/P01'][0]))[:45]
    if len(rows)!=45:raise ValueError('registered45real P01windows unavailable')
    with np.load(ws/'draft_off/weights.npz',allow_pickle=False) as source:z={k:source[k].copy() for k in source.files}
    reference=owning_mix();cases={}
    for mode in ('known','unrecognized'):
        baseline=instrument(load(ws/'draft_base','original321_'+mode,channels),mode=='unrecognized')
        off=instrument(load(ws/'draft_off','disabled321_'+mode,channels),mode=='unrecognized')
        on=instrument(load(ws/'draft_jsall','active321_'+mode,channels),mode=='unrecognized')
        maximum=0.;changed_before10=0;changed_after_rec=0;delta_fbl=0.;delta_fb=0.;off_scores=[];on_scores=[]
        for t,r in enumerate(rows):
            a,pa,ta=frame(baseline,x[r]);b,pb,tb=frame(off,x[r]);c,pc,tc=frame(on,x[r])
            if a!=b or not np.array_equal(pa,pb):raise ValueError('disabled generated bundle changed final predict/output')
            for field in ('mix','shadow','core1','u','out_v5'):
                if not np.array_equal(ta[field],tb[field]) or not np.array_equal(ta[field],tc[field]):
                    raise ValueError('original inner path changed: '+field)
            for field in ('contrast','identity'):
                if not same_gate(ta[field],tc[field]):raise ValueError('original gate changed')
            if baseline.ft.lab!=on.ft.lab or not np.array_equal(baseline.ft.wt,on.ft.wt):
                raise ValueError('original fine-tuning label chain changed')
            expected,share=reference_mix(on,tc,z,reference)
            error=float(np.abs(expected-tc['v6']['mix6']).max());maximum=max(maximum,error)
            if error>1e-10 or abs(share-1/17)>1e-15:raise ValueError('active outer mix differs from owning study recipe')
            for slot in ('fb','fbl'):
                if slot in tc['v6']['p_ft']:
                    delta=float(np.abs(tc['v6']['p_ft'][slot]-tc['v6']['members'][on.ft.member[slot]]).max())
                    if slot=='fbl':delta_fbl=max(delta_fbl,delta)
                    else:delta_fb=max(delta_fb,delta)
            changed_before10+=int(t<10 and not np.array_equal(pa,pc))
            changed_after_rec+=int(on.recB and not np.array_equal(pa,pc))
            off_scores.append(pa);on_scores.append(pc)
        if on.js_calls!=45 or off.js_calls!=0 or not np.array_equal(baseline.w,on.w):
            raise ValueError('member-use/original weight guard failed')
        if baseline.log6['rec']!=on.log6['rec'] or baseline.log6['err'] or on.log6['err']:
            raise ValueError('recognition or runtime error guard failed')
        if mode=='unrecognized' and (on.recB or delta_fbl<=1e-7 or delta_fb<=1e-7):
            raise ValueError('unrecognized fine-tuning path inactive')
        # Reset and interleaved independent copies, using genuine45window source prefix.
        on.reset_state();left,right=copy.deepcopy(on),copy.deepcopy(on)
        prefix=[]
        for t in range(17):
            prefix.append(frame(left,x[rows[t]])[1])
            if t<5:frame(right,x[rows[-1-t]])
        if not np.array_equal(np.array(prefix),np.array(on_scores[:17])):
            raise ValueError('session reset/deepcopy/prefix differs')
        cases[mode]=dict(disabled_final_bitwise=True,inner_bitwise=True,original_ft_labels_bitwise=True,
                         owning_mix_max_error=maximum,route_share=on.routeB.a,recognized_at=baseline.log6['rec'],
                         active_before10_changed_rows=changed_before10,active_after_rec_changed_rows=changed_after_rec,
                         fbl_ft_delta=delta_fbl,old_fb_ft_delta_excluded=delta_fb,cc_calls=45,reset_deepcopy_prefix=True,
                         cp_fused=baseline.cp.fused,scope='engineering in-sample components;no accuracy')
    result=dict(status='generated_bundle_active_engineering_guards_passed',cases=cases,seconds=time.monotonic()-start,
                source_sha256=sha(ws/'source.sha256'),build_sha256=sha(ws/'build.json'),
                original_inner_gate_identity=True,original_ft_label_identity=True,disabled_identity=True,
                study_outer_mix_parity=True,route_share17_verified=True,
                limitations='P01CC30heldfold;other original members all10;no heldwhole/returning/lateCP/worker/timing claim')
    (ws/'outer_active_path.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
