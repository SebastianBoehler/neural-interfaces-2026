"""New phase-matched validation; previous GPU bitwise reset failures remain failures."""
import copy
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

import guard_shared as common
from phase_state import neural_state,rng
from record import load_model
from reference import sha
from shared_raw import verify_raw

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
VARIANTS={'fir':ROOT/'ni26_js321_20261117_shared2','reuse':ROOT/'ni26_js321_20261117_inputreuse2'}
CONTROL=ROOT/'ni26_js321_20261117_ftrepeat2/ft_repeat.json'


def load(path,tag,channels,device='cuda',unknown=False):
    m=load_model(path,channels,tag,device)
    if Path(type(m).step.__globals__['__file__']).resolve()!=(path/'submission.py').resolve():raise ValueError('module path mismatch')
    m.cp.original_step=m.cp.step;m.cp.step=types.MethodType(common.capture,m.cp)
    if unknown:m.route.contrast.step=lambda *a:None;m.route.identity.step=lambda *a:None
    return m


def zero(m):
    state=neural_state(m)
    if not state['all_stream_buffers_zero'] or any(state['original_input_n']) or any(state[k] for k in ('core1_n','core2_n','auxiliary_input_n','eye_n','rest_input_n','ft_labels','cp_t')):
        raise ValueError('registered model reset state is nonzero')
    return state


def check_pair(old,new,x,device,bitwise=False):
    maximum=0.;saved=[];initial=[neural_state(m) for m in (old,new)]
    for t,value in enumerate(x):
        a,pa,ta=common.frame(old,value);b,pb,tb=common.frame(new,value)
        error=float(abs(pa-pb).max());maximum=max(maximum,error)
        if a!=b or (not np.array_equal(pa,pb) if bitwise else error>1e-8):raise ValueError('phase-matched old/optimized output parity failed')
        for key in ('core1','mix','u','shadow','out_v5'):
            if not np.array_equal(ta[key],tb[key]):raise ValueError('original inner changed: '+key)
        for key in ('contrast','identity'):
            if not (ta[key] is None and tb[key] is None) and not np.array_equal(ta[key],tb[key]):raise ValueError('original gate changed')
        if not np.array_equal(old.ft.lab,new.ft.lab) or not np.array_equal(old.ft.wt,new.ft.wt):raise ValueError('original FT labels changed')
        for slot in old.ft.X:
            if len(old.ft.X[slot])!=len(new.ft.X[slot]) or (old.ft.X[slot] and not np.array_equal(old.ft.X[slot][-1],new.ft.X[slot][-1])):
                raise ValueError('original FT input buffer changed')
        if old.ft.gens is not None:
            a={s:hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest() for s,g in old.ft.gens.items()}
            b={s:hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest() for s,g in new.ft.gens.items()}
            if a!=b:raise ValueError('augmentation generator states differ')

        if getattr(new,'js_reuse',False):
            if new.js_step!=t+1 or new.js_done!=t+1 or new.js_bank.groups[0][0].n!=0 or not np.array_equal(new.nets.last_inputs[35],old.nets.last_inputs[35]):
                raise ValueError('once/current/readonly-input check failed')
        saved.append((pa,pb))
    final=[neural_state(m) for m in (old,new)]
    if any(a['immutable_sha256']!=b['immutable_sha256'] for a,b in zip(initial,final)):raise ValueError('original/aux neural immutable tensors changed')
    if old.log6['err'] or new.log6['err'] or old.cp.log['err'] or new.cp.log['err']:raise ValueError('phase runtime error')
    return maximum,saved,initial,final


def reset_pair(old,new,x,saved,device):
    old.reset_state();new.reset_state();reset=[zero(m) for m in (old,new)]
    maximum,warm,_,_=check_pair(old,new,x[:17],device)
    cold_drift=[max(float(abs(p[k]-q[k]).max()) for p,q in zip(saved[:17],warm)) for k in (0,1)]
    if device=='cpu' and any(cold_drift):raise ValueError('CPU self reset is not bitwise')
    old.reset_state();new.reset_state();left=[copy.deepcopy(m) for m in (old,new)];right=[copy.deepcopy(m) for m in (old,new)]
    for m in left+right:zero(m)
    for i,value in enumerate(x[:17]):
        outputs=[common.frame(m,value)[1] for m in left]
        if float(abs(outputs[0]-outputs[1]).max())>1e-8:raise ValueError('paired GPU deepcopy prefix differs')
        for k,p in enumerate(outputs):
            different=(not np.array_equal(p,warm[i][k])) if device=='cpu' else float(abs(p-warm[i][k]).max())>1e-8
            if different:raise ValueError('same-phase self deepcopy/interleaving differs')
        if i<5:
            for m in right:common.frame(m,x[-1-i])
    return dict(max_paired_reset_error=maximum,cold_to_reset_drift=cold_drift,reset_states=reset,deepcopy_interleaved=True,
                cpu_self_bitwise=device=='cpu',gpu_self_bitwise_not_required_old_failure_preserved=device=='cuda')


def main(variant):
    start=time.monotonic();torch.set_num_threads(1)
    if variant not in VARIANTS or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('registered variant/A100 required')
    folder=VARIANTS[variant];control=json.loads(CONTROL.read_text());complete=json.loads((folder/'complete.json').read_text())
    if control['status'] not in ('original_JS215_FT_selfrepeat_characterized','deterministic_original215_control_passed') or control['ft_refits']!=[20,20] or control['cp_fused']!=[125,125]:raise ValueError('complete original FT215 control required first')
    if control['status']=='deterministic_original215_control_passed' and not torch.are_deterministic_algorithms_enabled():raise ValueError('controlled executor flag missing')
    if control['old_weights_sha256']!=sha(common.OLD/'weights.npz'):raise ValueError('original control weights changed')
    for name,digest in complete['files'].items():
        if sha(folder/'prepared_jsall'/name)!=digest or (name!='submission.py' and digest!=sha(common.OLD/name)):raise ValueError('same-weight artifact mismatch')
    with np.load(common.DATA,allow_pickle=False) as z:x=z['X'][:215].copy();channels=z['ch_names']
    before=rng();probe=load(folder/'prepared_jsall','raw_phase_'+variant,channels)
    raw=verify_raw(type(probe).step.__globals__,'cuda');del probe;torch.cuda.empty_cache();cases={}
    for mode,n in (('known',45),('unrecognized',215)):
        old=load(common.OLD,'old_phase_'+mode+variant,channels,unknown=mode=='unrecognized')
        new=load(folder/'prepared_jsall','new_phase_'+mode+variant,channels,unknown=mode=='unrecognized')
        zero(old);zero(new);maximum,saved,initial,final=check_pair(old,new,x[:n],'cuda')
        activity=dict(old_ft=old.ft.refits,new_ft=new.ft.refits,old_cp=old.cp.refits,new_cp=new.cp.refits,old_fused=old.cp.fused,new_fused=new.cp.fused)
        if mode=='unrecognized' and (new.recB or new.ft.refits!=20 or new.cp.refits!=10 or new.cp.fused!=125):raise ValueError('actual late FT/CP activation failed')
        reset=reset_pair(old,new,x,saved,'cuda')
        original=load(common.BASE,'original_off_'+mode+variant,channels,unknown=mode=='unrecognized')
        off=load(folder/'draft_off','new_off_'+mode+variant,channels,unknown=mode=='unrecognized')
        for value in x[:45]:
            a,pa,_=common.frame(original,value);b,pb,_=common.frame(off,value)
            if a!=b or not np.array_equal(pa,pb):raise ValueError('new disabled final output changed')
        cases[mode]=dict(max_output_error=maximum,original_inner_labels_bitwise=True,off45_bitwise=True,initial_states=initial,final_states=final,
                         activity=activity,reset=reset,input_reuse_checked=variant=='reuse')
        del old,new,original,off;torch.cuda.empty_cache()
    old=load(common.OLD,'cpu_old_'+variant,channels,'cpu');new=load(folder/'prepared_jsall','cpu_new_'+variant,channels,'cpu')
    maximum,saved,_,_=check_pair(old,new,x[:17],'cpu')
    cpu=reset_pair(old,new,x,saved,'cpu')
    report=dict(status='phase_matched_optimization_guard_passed',variant=variant,artifact=str(folder),raw=raw,cases=cases,cpu=cpu,
                cpu_paired_max_error=maximum,control_sha256=sha(CONTROL),original_FT_selfrepeat_max=control['max_errors'],
                complete_sha256=sha(folder/'complete.json'),artifact_source_sha256=sha(folder/'source.sha256'),source_sha256=sha(Path('source.sha256')),
                rng_before=before,rng_after=rng(),seconds=time.monotonic()-start,accuracy_computed=False,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                old_GPU_bitwise_reset_failures_preserved=True,scope='phase-matched engineering;not a timing-bar/returning/held-accuracy pass')
    Path('phase_'+variant+'.json').write_text(json.dumps(report,indent=2));print(report['status'],variant,flush=True)


if __name__=='__main__':main(sys.argv[1])
