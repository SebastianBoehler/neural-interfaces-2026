"""Prospective FT/CP-warmed counterbalanced production timing; no accuracy."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import worker


def checked(model,x,device):
    row,cur=worker.run(model,x,device)
    row['activity']=worker.activity(cur,len(x),'unrecognized')
    if cur.route.log['err'] or cur.routeB.log['err'] or cur.cf.log['err']:
        raise ValueError('original downstream runtime error')
    if hasattr(cur,'js_calls') and cur.js_calls!=len(x):
        raise ValueError('auxiliary member inactive')
    row['label_sha256']=hashlib.sha256(np.asarray(row.pop('labels'),np.int64).tobytes()).hexdigest()
    return row


def main():
    functional=json.loads(Path('/functional/functional_cuda.json').read_text())
    if functional['status']!='complete_functional_only_matrix_passed_NOT_timing_qualified' or functional['variant']!='reuse':
        raise ValueError('fixed finalist full functional prerequisite absent')
    if torch.are_deterministic_algorithms_enabled():raise ValueError('standard executor required')
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('A100 required')
    torch.set_num_threads(1)
    complete=json.loads(Path('/experiment/complete.json').read_text())
    for name,digest in complete['files'].items():
        if worker.sha(Path('/experiment/prepared_jsall')/name)!=digest:raise ValueError('artifact changed')
    with np.load('/input.npz',allow_pickle=False) as z:x=z['X'].copy();channels=z['ch_names']
    models={key:worker.load_model(path,channels,'warm321_'+key,'cuda') for key,path in
            (('base','/active/draft_base'),('js','/experiment/prepared_jsall'))}
    for model in models.values():
        model.route.contrast.step=lambda *a:None
        model.route.identity.step=lambda *a:None
    cold={key:checked(model,x[:800],'cuda') for key,model in models.items()}
    rows=[]
    for n in (190,378,800):
        for repeat,order in enumerate((('base','js'),('js','base'),('base','js'),('js','base'))):
            values={key:checked(models[key],x[:n],'cuda') for key in order}
            rows.append(dict(windows=n,repeat=repeat,first=order[0],**values,
                             ratio=values['js']['seconds']/values['base']['seconds']))
            print(n,repeat,rows[-1]['ratio'],flush=True)
    limit=(60/1.3)/43.5
    report=dict(status='warm_counterbalanced_complete',variant='reuse',cold_setup=cold,rows=rows,
                original_bar=limit,all_warmed_ratios_under_original_bar=all(r['ratio']<=limit for r in rows),
                torch=torch.__version__,cuda=torch.version.cuda,threads=torch.get_num_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                complete_sha256=worker.sha('/experiment/complete.json'),accuracy_computed=False,
                scope='paired warm engineering;old first-stop failures preserved;not sealed-time proof')
    Path('/out/warm_timing.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
