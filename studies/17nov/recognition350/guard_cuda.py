"""Real-window A100 semantics; no task labels or metrics are loaded."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from returning_pair_guard import ROOT,models,zero,state
from returning_pair_checks import Frames,compare
from package_guard import instrument,frame,owning_mix,same_gate
from record import load_model,ft_clock_restart
from benchmark_utils.streaming import fresh_model
from returning_assets import sha


def main():
    start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if torch.__version__!='2.14.1+cu130' or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise ValueError('explicit clean A100 required')
    import os
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('prospective deterministic CUDA workspace required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    with np.load(ROOT/'neural-interfaces-2026/data/proteus_windows.npz',allow_pickle=True) as d:
        X=d['X'].copy();subject=d['subject'].copy();session=d['session'].astype(str);channels=d['ch_names'].copy()
    source=np.flatnonzero((subject=='Dreyer2026Proteus/P04')&(session=='1'))
    cases={};reference=owning_mix()
    for name,rows,unknown in [('known80',source[:80],False),('forced_unrecognized215',np.resize(source,215),True)]:
        current=models(channels,'P04_s23',unknown,'cuda')
        for mode in (1,2):
            path=Path('/probe/bundles/P04_s23')/str(mode)
            current['on' if mode==1 else 'policy']=instrument(load_model(path,channels,'policy350_'+name+str(mode),'cuda'),unknown)
        # The original off arm is replaced by the new mode0, preserving the baseline identity check.
        current['off']=instrument(load_model(Path('/probe/bundles/P04_s23/0'),channels,'policy350_'+name+'0','cuda'),unknown)
        before={k:zero(m)['immutable_sha256'] for k,m in current.items()}
        for m in current.values():
            for r in source[:17]:frame(m,X[r])
            m.reset_state();zero(m);ft_clock_restart(m)
        runner=Frames(current,'cuda');saved={k:[] for k in current};error=0.
        with np.load(ROOT/'ni26_js321_20261117_returning_bundle2/P04_s23/js/weights.npz') as z:weights={k:z[k].copy(order='K') for k in z.files}
        for t,r in enumerate(rows):
            output=runner(current,X[r]);compare({k:current[k] for k in ('base','off','js')},{k:output[k] for k in ('base','off','js')},t,weights,reference)
            for k in current:saved[k].append(output[k][1])
            if (t+1)%20==0:print(json.dumps(dict(case=name,windows=t+1,seconds=time.monotonic()-start,activity={k:dict(rec=m.log6['rec'],FT=m.ft.refits,CP=m.cp.fused) for k,m in current.items()})),flush=True)
            for k in ('on','policy'):
                m=current[k];trace=output[k][2];ref=output['base'][2]
                for field in ('core1','mix','u','shadow','out_v5'):
                    if not np.array_equal(trace[field],ref[field]):raise ValueError('original inner mismatch '+field)
                if any(not same_gate(trace[f],ref[f]) for f in ('contrast','identity')):raise ValueError('gate changed')
                if not np.array_equal(m.ft.lab,current['base'].ft.lab) or not np.array_equal(m.ft.wt,current['base'].ft.wt):raise ValueError('FT labels changed')
                for slot in m.ft.X:
                    if len(m.ft.X[slot])!=len(current['base'].ft.X[slot]) or (m.ft.X[slot] and not np.array_equal(m.ft.X[slot][-1],current['base'].ft.X[slot][-1])):raise ValueError('FT input changed')
                if m.js_calls!=t+1 or m.js_done!=t+1 or m.js_step!=t+1 or m.js_bank.groups[0][0].n:raise ValueError('auxiliary callback state changed')
                if not np.array_equal(m.nets.last_inputs[35],current['base'].nets.last_inputs[35]):raise ValueError('raw input changed')
                if k=='on' or unknown:
                    diff=float(np.abs(output[k][1]-output['js'][1]).max());error=max(error,diff)
                    if diff>1e-8 or output[k][0]!=output['js'][0]:raise ValueError('all-on/unrecognized JS identity failed')
                expected_js=(k=='on' or not m.recB)
                if not np.isclose(m.routeB.a,1/17 if expected_js else 1/16,rtol=0,atol=1e-15):raise ValueError('active denominator changed')
                if any((m.log6['err'],m.cp.dead,m.cp.log['err'],m.dead,m.ft.over)):raise ValueError('runtime or budget guard failed')
        active={k:dict(rec=m.log6['rec'],ft=m.ft.refits,cp=m.cp.fused,js=getattr(m,'policy350_js',None),base=getattr(m,'policy350_base',None)) for k,m in current.items()}
        policy=current['policy']
        if unknown:
            if any(v['rec']>=0 or v['ft']!=20 or v['cp']!=125 for v in active.values()) or policy.policy350_js!=215 or policy.policy350_base:raise ValueError('unrecognized/late FT CP path inactive')
        elif policy.log6['rec']!=10 or policy.policy350_js!=10 or policy.policy350_base!=70:raise ValueError('actual known transition inactive')
        for k,m in current.items():
            if state(m)['immutable_sha256']!=before[k]:raise ValueError('immutable model buffers changed')
            m.reset_state();zero(m);left,right=fresh_model(m),fresh_model(m);ft_clock_restart(left);ft_clock_restart(right)
            for i,r in enumerate(rows[:17]):
                if not np.array_equal(frame(left,X[r])[1],saved[k][i]):raise ValueError('fresh reset prefix differs')
                frame(right,X[rows[-1-i]])
        cases[name]=dict(activation=active,maximum_identity_error=error,original_inner_gates_FT_exact=True,all_buffers_immutable=True,reset_deepcopy_prefix_exact=True)
        Path('/out/cases_partial.json').write_text(json.dumps(dict(status='PARTIAL_CASES_NOT_FULL_PASS',cases=cases,source_sha256=sha('/probe/source.sha256'),accuracy_computed=False),indent=2))
        print(name+' PASS '+str(time.monotonic()-start),flush=True)
    if torch.cuda.max_memory_allocated()==0:raise ValueError('CUDA executor was not active')
    Path('/out/guard.json').write_text(json.dumps(dict(status='POLICY350_GPU_SOURCE_FUNCTIONAL_PASS_NOT_ACCURACY',cases=cases,seconds=time.monotonic()-start,source_sha256=sha('/probe/source.sha256'),build_sha256=sha('/probe/build.json'),executor='cleanSIF_A100_deterministic',CPU_unknown215_proven=False,CUDA_device=torch.cuda.get_device_name(),CUDA_peak_allocated=torch.cuda.max_memory_allocated(),torch_deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],accuracy_computed=False),indent=2))


if __name__=='__main__':main()
