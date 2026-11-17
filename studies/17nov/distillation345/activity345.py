"""Direct KL-only native-body gradients on the actual source batch and honest OOF targets."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from common345 import source,native,capture,sha,digest,T,BASE

root=Path(os.environ['DISTILL_ROOT']);torch.set_num_threads(6)
if Path(T.__file__).resolve()!=(BASE/'experiments/ts_core2.py').resolve():raise ValueError('original Python core import path')
troot=Path(os.environ['TEACHER_ROOT']);card=json.loads((troot/'teacher.json').read_text())
if sha(troot/'teacher.npz')!=card['teacher_sha256']:raise ValueError('teacher changed')
d,fit,held,people,y=source();records={}
for arm,key in (('KD','proba'),('deranged','deranged')):
    model=native(d);value=capture(model,fit);model.net.train()
    j=model._batches(value['rows'],np.random.default_rng(0))[0];r=model._order[j]
    with np.load(troot/'teacher.npz') as z:
        if not np.array_equal(z['index'],fit):raise ValueError('OOF scope')
        tp=torch.from_numpy(np.clip(z[key][np.searchsorted(fit,r)],1e-8,1)**.5).float();tp/=tp.sum(1,keepdim=True)
    logits=model.net(value['X'][j]);loss=torch.nn.functional.kl_div(torch.log_softmax(logits/2,1),tp,reduction='batchmean')*4
    grads=torch.autograd.grad(loss,[p for p in model.net.parameters() if p.requires_grad],allow_unused=True)
    body=[];head=[]
    for (name,p),g in zip(((n,p) for n,p in model.net.named_parameters() if p.requires_grad),grads):
        if g is not None:
            if not torch.isfinite(g).all():raise ValueError('nonfinite KL-only gradient')
            (head if 'final_layer' in name else body).append(float(g.square().sum()))
    if not body or not head or sum(body)<=0 or sum(head)<=0:raise ValueError('KL body/head gradient inactive')
    records[arm]=dict(body_gradient_norm=float(np.sqrt(sum(body))),head_gradient_norm=float(np.sqrt(sum(head))),KL=float(loss.detach()),batch_rows_sha256=digest(r))
report=dict(passed=True,records=records,teacher_sha256=card['teacher_sha256'],source_sha256=sha(root/'source.sha256'),
    module_paths=dict(core=str(T.__file__),native_class=model.__class__.__module__),core_sha256=sha(T.__file__),held_labels_opened=False)
(root/'activity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
