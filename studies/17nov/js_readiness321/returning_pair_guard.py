"""Actual private source-bridge module/off/inner/FT/CP/reset guards; no query scores."""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
sys.path.insert(0,str(ROOT/'neural-interfaces-2026/2026-competition/tracks/bci_decoding'))
import benchmark_utils.base_solver
from benchmark_utils.streaming import fresh_model
from record import load_model,ft_clock_restart
from package_guard import instrument,frame,reference_mix,owning_mix,same_gate
from phase_state import neural_state
from returning_pair_checks import Frames,compare
from returning_assets import sha

BUNDLE=ROOT/'ni26_js321_20261117_returning_bundle2'


def state(m):
    if not hasattr(m,'js_bank'):m.js_bank=None  # Observer-only sentinel; original baseline has no auxiliary bank.
    return neural_state(m)


def zero(m):
    s=state(m)
    if not s['all_stream_buffers_zero'] or any(s['original_input_n']) or any(s[k] for k in ('core1_n','core2_n','auxiliary_input_n','eye_n','rest_input_n','ft_labels','cp_t')):raise ValueError('actual fresh/reset state nonzero')
    return s


def models(channels,fold,unknown,device):
    result={}
    for arm in ('base','off','js'):
        path=BUNDLE/fold/arm;m=load_model(path,channels,'source_bridge_'+fold+arm,device)
        if Path(type(m).step.__globals__['__file__']).resolve()!=(path/'submission.py').resolve():raise ValueError('private module path differs')
        result[arm]=instrument(m,unknown)
    return result


def preflight():
    if torch.__version__!='2.14.1+cu130' or 'neural-interfaces-2026/.venv/' in Path('/proc/self/maps').read_text():raise ValueError('actual clean worker image required')
    torch.set_num_threads(1);proof=json.loads((BUNDLE/'bundle_complete.json').read_text())
    if proof['status']!='twofold_three_arm_private_source_bridge_built_NOT_validated':raise ValueError('private owning build absent')
    for a in proof['artifacts']:
        if sha(BUNDLE/a['path']/'submission.py')!=a['module_sha256']:raise ValueError('private source module changed')
        for name,digest in a['files'].items():
            if sha(BUNDLE/a['path']/name)!=digest:raise ValueError('private source asset changed')
    with np.load(ROOT/'neural-interfaces-2026/data/proteus_windows.npz',allow_pickle=True) as d:channels=d['ch_names'].copy()
    for fold in ('P04_s23','P10_s23'):
        for arm,m in models(channels,fold,False,'cpu').items():
            zero(m)
            kinds={i:type(net).__name__ for _,members in m.nets.groups for i,net in members}
            if any(kinds[i]!='Shallow' for i in range(10)) or any(kinds[i]!='FBC' for i in range(20,24)):raise ValueError('original plain inference classes differ')
    print('ACTUAL_SIF_BOTH_FOLDS_THREE_MODULES_LOAD_ZERO_CLASSES_OK',flush=True)


def main():
    start=time.monotonic();torch.set_num_threads(1)
    if torch.__version__!='2.14.1+cu130' or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise ValueError('actual worker A100 required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('prospective deterministic CUDA workspace absent')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    with np.load(ROOT/'neural-interfaces-2026/data/proteus_windows.npz',allow_pickle=True) as d:X=d['X'].copy();sub=d['subject'].copy();ses=d['session'].astype(str);channels=d['ch_names'].copy()
    known=np.flatnonzero((sub=='Dreyer2026Proteus/P04')&(ses=='1'))[:80]
    held=np.flatnonzero((sub=='Dreyer2026Proteus/P04')&(ses=='2'));stress=np.resize(held,215)
    cases={};reference=owning_mix()
    for mode,rows,unknown in (('source_known80',known,False),('forced_unrecognized215_real_window_replay',stress,True)):
        template=models(channels,'P04_s23',unknown,'cuda');before={k:zero(m) for k,m in template.items()}
        for m in template.values():
            for r in known[:17]:frame(m,X[r])
            m.reset_state();zero(m)
        cur={k:fresh_model(m) for k,m in template.items()}
        for m in cur.values():ft_clock_restart(m);zero(m)
        with np.load(BUNDLE/'P04_s23/js/weights.npz') as z:weights={k:z[k].copy(order='K') for k in z.files}
        maximum=0.;active=[];saved={k:[] for k in cur}
        frames=Frames(cur)
        for t,r in enumerate(rows):
            data=frames(cur,X[r]);error,activity,_=compare(cur,data,t,weights,reference)
            maximum=max(maximum,error);active.append(activity)
            for k in cur:saved[k].append(data[k][1])
        activation={k:dict(rec=m.log6['rec'],ft_refits=m.ft.refits,cp_fused=m.cp.fused,cp_stopped=m.cp.stopped) for k,m in cur.items()}
        for k,m in cur.items():
            if state(m)['immutable_sha256']!=before[k]['immutable_sha256']:raise ValueError('original/auxiliary weights or non-session buffers changed')
            if unknown and (m.log6['rec']>=0 or m.ft.refits<=0 or m.cp.fused<=0):raise ValueError('forced unknown/FT/lateCP inactive')
            if not unknown and (m.log6['rec']!=10 or m.ft.refits or m.cp.stopped!=10):raise ValueError('source known first recognition differs from10')
            m.reset_state();zero(m);left,right=fresh_model(m),fresh_model(m);ft_clock_restart(left);ft_clock_restart(right)
            prefix=[]
            for i,r in enumerate(rows[:17]):prefix.append(frame(left,X[r])[1]);frame(right,X[rows[-1-i]])
            if not np.array_equal(np.asarray(prefix),np.asarray(saved[k][:17])):raise ValueError('source reset/deepcopy/prefix differs')
        if max(active[:10])<=1e-12 or (not unknown and max(active[10:])<=1e-12):raise ValueError('JS source activity before/after recognition absent')
        cases[mode]=dict(off_base_bitwise=True,inner_gate_FT_chain_bitwise=True,owning_mix_maxabs=maximum,source_JS_activity_max=max(active),
                         records=activation,reset_copy_prefix_bitwise=True,
                         scope='source known session or repeated-real-window functional stress;NOaccuracy')
        del cur,template;torch.cuda.empty_cache()
    Path('/out/pair_guard.json').write_text(json.dumps(dict(status='actual_private_source_bridge_guards_passed_NOT_accuracy',cases=cases,
        source_sha256=sha('source.sha256'),bundle_manifest_sha256=sha(BUNDLE/'bundle_complete.json'),seconds=time.monotonic()-start,
        deterministic_validation_executor=True,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],torch_deterministic=torch.are_deterministic_algorithms_enabled(),
        accuracy_computed=False,scope='ENGINEERING_CONDITIONAL;originalrest_eye/classical_production_exposure_retained;notstandard_worker_timing'),indent=2))


if __name__=='__main__':preflight() if sys.argv[1]=='preflight' else main()
