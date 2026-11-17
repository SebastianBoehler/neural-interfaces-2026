"""Separate complete functional-only matrix; descriptive timings never rescue an earlier timing-bar failure."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

import worker
from worker_phase import main as unused_timing_entry


def main(variant,device):
    if variant not in ('fir','reuse') or device not in ('cpu','cuda'):raise ValueError('registered variant/device required')
    r=json.loads(Path('/semantics/phase_'+variant+'.json').read_text())
    if r['status']!='phase_matched_optimization_guard_passed' or r['complete_sha256']!=worker.sha('/experiment/complete.json'):
        raise ValueError('controlled semantic/artifact proof missing')
    if torch.are_deterministic_algorithms_enabled():raise ValueError('functional study uses unchanged standard production executor')
    torch.set_num_threads(1)
    if device=='cuda' and (not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0)):raise ValueError('explicit A100 required')
    complete=json.loads(Path('/experiment/complete.json').read_text())
    for name,digest in complete['files'].items():
        if worker.sha(Path('/experiment/prepared_jsall')/name)!=digest:raise ValueError('artifact changed')
    with np.load('/input.npz',allow_pickle=False) as z:x=z['X'].copy();channels=z['ch_names']
    models={name:worker.load_model(path,channels,'functional321_'+variant+device+name,device) for name,path in
            (('base','/active/draft_base'),('off','/experiment/draft_off'),('js','/experiment/prepared_jsall'))}
    a,_=worker.run(models['base'],x[:17],device);b,_=worker.run(models['off'],x[:17],device)
    if a['labels']!=b['labels']:raise ValueError('functional disabled prefix differs')
    a,_=worker.run(models['js'],x[:17],device);b,_=worker.run(models['js'],x[:17],device)
    if a['labels']!=b['labels']:raise ValueError('functional reset prefix differs')
    rows=[]
    for mode in ('known','unrecognized'):
        pair={name:copy.deepcopy(models[name]) for name in ('base','js')}
        if mode=='unrecognized':
            for m in pair.values():m.route.contrast.step=lambda *a:None;m.route.identity.step=lambda *a:None
        for m in pair.values():worker.run(m,x[800:817],device)
        for i,n in enumerate((190,378,800)):
            values={};order=('base','js') if i%2==0 else ('js','base')
            for name in order:
                values[name],m=worker.run(pair[name],x[:n],device);values[name]['activity']=worker.activity(m,n,mode)
                if m.route.log['err'] or m.routeB.log['err'] or m.cf.log['err']:raise ValueError('original routing/filter runtime error')
                if name=='js' and m.js_calls!=n:raise ValueError('auxiliary member inactive')
                values[name]['label_sha256']=__import__('hashlib').sha256(np.array(values[name].pop('labels'),np.int64).tobytes()).hexdigest()
            rows.append(dict(mode=mode,windows=n,first=order[0],**values,descriptive_ratio=values['js']['seconds']/values['base']['seconds']))
            print('functional',variant,device,mode,n,'passed activity',flush=True)
    maps=Path('/proc/self/maps').read_text()
    if 'neural-interfaces-2026/.venv/' in maps:raise ValueError('worker loaded host numerical library')
    result=dict(status='complete_functional_only_matrix_passed_NOT_timing_qualified',variant=variant,device=device,rows=rows,
                torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                prior_timing_failures_preserved=True,old_bar_reinterpreted=False,accuracy_computed=False,
                scope='all190/378/800known/unrecognizedengineering;timingsdescriptive;notwhole-limit/held/returning/promotion proof')
    Path('/out/functional_'+device+'.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
