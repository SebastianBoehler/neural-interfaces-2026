"""Conditional same-weight component profile; no optimized package or accuracy."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,'/official/tracks/bci_decoding')
import benchmark_utils.base_solver
from benchmark_utils.streaming import fresh_model
from record import load_model


def sync():torch.cuda.synchronize()


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BankTimer:
    def __init__(self,bank):self.bank=bank;self.times=[]
    def __getattr__(self,name):return getattr(self.bank,name)
    def reset_state(self):self.bank.reset_state();self.times=[]
    def __call__(self,x):
        sync();start=time.perf_counter();result=self.bank(x);sync()
        self.times.append(time.perf_counter()-start);return result


def run_nets(nets,inputs,shared):
    for net in nets:net.reset_state()
    values=[];sync();started=time.perf_counter()
    with torch.no_grad():
        for value in inputs:
            cache={};values.append(torch.stack([net.proba(value,cache if shared else {}) for net in nets]).cpu().numpy())
    sync();return np.array(values),time.perf_counter()-started


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('explicit A100 required')
    r=json.loads(Path('/experiment/complete.json').read_text());root=Path('/experiment/prepared_jsall')
    if r['status']!='six_seed_all10_complete_private_copy_prepared':raise ValueError('incomplete source artifact')
    for name,want in r['files'].items():
        if digest(root/name)!=want:raise ValueError('artifact changed')
    with np.load('/input.npz',allow_pickle=False) as z:x=z['X'][:17].copy();channels=z['ch_names']
    model=load_model(root,channels,'profile321','cuda');g=type(model).step.__globals__
    with np.load(root/'weights.npz',allow_pickle=False) as z:
        p={k:z['js_'+k].copy(order='K') for k in ('align','k','G','scale','shrink')}
        original={k:z['net35_'+k].copy(order='K') for k in p};files=[str(f) for f in z['js_cc_files']]
    if g['input_key'](p)!=g['input_key'](original):raise ValueError('auxiliary/native input groups differ')
    stream=g['InputStream'](p);start=time.perf_counter();aligned=[stream(value) for value in x]
    input_seconds=time.perf_counter()-start;inputs=[torch.from_numpy(v[None]).cuda() for v in aligned]
    full=[net for _,members in model.js_bank.groups for _,net in members]
    shared=[g['SharedFIRSJit'](dict(p,sjit_file=np.array(name)),'cuda',root) for name in files]
    if len(full)!=6 or len(shared)!=6 or any(type(n).__name__!='SJit' for n in full):raise ValueError('actual baseline auxiliary dispatch changed')
    # Warm then reset before each alternating paired same-weight component run.
    run_nets(full,inputs[:4],False);run_nets(shared,inputs[:4],True);runs=[]
    for round_index in range(2):
        outputs={};times={};order=('full','shared') if round_index==0 else ('shared','full')
        for name in order:
            outputs[name],times[name]=run_nets(full if name=='full' else shared,inputs,name=='shared')
        error=float(abs(outputs['full']-outputs['shared']).max())
        if error>1e-8 or not np.array_equal(outputs['full'].argmax(-1),outputs['shared'].argmax(-1)):
            raise ValueError('existing fingerprinted SharedFIR helper differs from the six current full traces')
        runs.append(dict(round=round_index,order=order,max_probability_error=error,seconds=times,ratio=times['shared']/times['full']))
    # Separate source component timing; instrumentation is not a new whole-system timing score.
    cur=fresh_model(model);cur.js_bank=BankTimer(cur.js_bank);cur.nets=BankTimer(cur.nets)
    tangent=cur._tangents;tangent_times=[]
    def timed_tangent(value):
        t=time.perf_counter();out=tangent(value);tangent_times.append(time.perf_counter()-t);return out
    cur._tangents=timed_tangent;steps=[]
    for value in x:
        sync();start=time.perf_counter();cur.step(value);sync();steps.append(time.perf_counter()-start)
    if cur.log6['err']:raise ValueError('instrumented known prefix error')
    sync();start=time.perf_counter()
    with torch.no_grad():
        for value in inputs:full[0].m.spectral_filtering(value)
    sync();fir_seconds=time.perf_counter()-start
    report=dict(status='same_weight_component_profile_complete',torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                source_artifact_sha256=digest('/experiment/complete.json'),baseline_dispatch=[type(n).__name__ for n in full],
                shared_dispatch=[type(n).__name__ for n in shared],fir_fingerprint=g['SBN_FIR_SHA256'],same_input_group=True,
                real_windows=17,input_seconds=input_seconds,single_fir_prefix_seconds=fir_seconds,microbenchmark=runs,
                median_step_seconds=float(np.median(steps)),median_aux_bank_seconds=float(np.median(cur.js_bank.times)),
                median_base_bank_seconds=float(np.median(cur.nets.times)),median_tangent_seconds=float(np.median(tangent_times)),
                scope='known17in-sample engineering components;not whole-adapter parity/held accuracy/worker-ready/timing-bar recovery',
                accuracy_computed=False,optimized_package_created=False)
    Path('/out/profile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
