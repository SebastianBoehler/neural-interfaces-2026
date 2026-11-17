"""Worker-image functional/state and paired timing probe; no labels or accuracy."""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,'/official/tracks/bci_decoding')
import benchmark_utils.base_solver
from benchmark_utils.streaming import fresh_model
from record import load_model,ft_clock_restart


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(device):
    if device=='cuda':torch.cuda.synchronize()


def run(model,x,device):
    sync(device);start=time.perf_counter();cur=fresh_model(model);ft_clock_restart(cur);out=[];calls=[]
    for window in x:
        sync(device);called=time.perf_counter();out.append(int(cur.predict(torch.from_numpy(window[None]))[0]));sync(device)
        calls.append(time.perf_counter()-called)
    sync(device)
    if cur.log6['err']:
        raise ValueError('runtime addition error')
    return dict(seconds=time.perf_counter()-start,cold_b1_seconds=calls[0],warm_b1_median=float(np.median(calls[1:])),labels=out,recognized_at=cur.log6['rec'],
                js_calls=cur.js_calls if hasattr(cur,'js_calls') else 0,ft_refits=cur.ft.refits if hasattr(cur.ft,'refits') else None,
                cp_fused=cur.cp.fused if cur.cp is not None else None),cur


def main(device):
    if device not in ('cpu','cuda'):raise ValueError('explicit device required')
    torch.set_num_threads(1)
    if device=='cuda' and (not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0)):
        raise ValueError('worker CUDA route requires A100')
    complete=json.loads(Path('/experiment/complete.json').read_text())
    if complete['status']!='six_seed_all10_complete_private_copy_prepared':raise ValueError('incomplete artifact')
    for name,digest in complete['files'].items():
        if sha(Path('/experiment/prepared_jsall')/name)!=digest:raise ValueError('artifact hash changed')
    with np.load('/input.npz',allow_pickle=False) as z:x=z['X'].copy();channels=z['ch_names']
    if x.shape[1:]!=(41,480) or len(x)<800:raise ValueError('real official tensor unavailable')
    models={k:load_model(path,channels,'worker321_'+device+k,device) for k,path in
            (('base','/active/draft_base'),('off','/active/draft_off'),('js','/experiment/prepared_jsall'))}
    # Actual independent-session copies; no module state may carry between them.
    b,_=run(models['base'],x[:17],device);o,_=run(models['off'],x[:17],device)
    if b['labels']!=o['labels']:raise ValueError('worker disabled path differs from original')
    j,_=run(models['js'],x[:17],device);repeat,_=run(models['js'],x[:17],device)
    if j['labels']!=repeat['labels'] or j['js_calls']!=17:raise ValueError('worker reset/repeat/member guard failed')
    rows=[];limit=(60/1.3)/43.5
    for mode in ('known','unrecognized'):
        pair={k:copy.deepcopy(models[k]) for k in ('base','js')}
        if mode=='unrecognized':
            for model in pair.values():model.route.contrast.step=lambda *a:None;model.route.identity.step=lambda *a:None
        for key in pair:run(pair[key],x[800:817],device)  # Untimed real warm prefix.
        for index,n in enumerate((190,378,800)):
            results={};order=('base','js') if index%2==0 else ('js','base')
            for key in order:
                results[key],cur=run(pair[key],x[:n],device)
                if key=='js' and cur.js_calls!=n:raise ValueError('worker auxiliary member became inactive')
            ratio=results['js']['seconds']/results['base']['seconds']
            rows.append(dict(mode=mode,windows=n,first=order[0],base=results['base'],js=results['js'],ratio=ratio))
            print(mode,n,ratio,flush=True)
            if ratio>limit:
                Path('/out/'+device+'_failure.json').write_text(json.dumps(dict(status='worker_slack_guard_failed',row=rows[-1],allowed_ratio=limit,no_later_rows_harvested=True),indent=2))
                raise ValueError('predeclared slack guard failed;stop before subsequent timing rows')
    maximum=max(r['ratio'] for r in rows);limit=(60/1.3)/43.5
    report=dict(status='worker_engineering_complete' if maximum<=limit else 'worker_slack_guard_failed',device=device,
                torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,rows=rows,max_ratio=maximum,
                allowed_ratio=limit,historical_minutes=43.5,hour_slack=60/(43.5*maximum),
                scope='real source windows concatenated only for length stress;not held accuracy or sealed forecast',
                policy='JS-all continues after recognition',accuracy_computed=False)
    maps=Path('/proc/self/maps').read_text()
    if 'neural-interfaces-2026/.venv/' in maps:raise ValueError('worker loaded host numerical libraries')
    Path('/out/'+device+'.json').write_text(json.dumps(report,indent=2))
    if maximum>limit:raise ValueError('predeclared runtime slack guard failed;no promotion')


if __name__=='__main__':main(sys.argv[1])
