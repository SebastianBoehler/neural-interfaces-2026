"""Prepared source-only eight-step native gradient/cost guard; requires verified336 API."""
import json
import os
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.parametrize import is_parametrized
from dependencies import sha,digest
from targets import TeacherHead

FIXTURE=Path('/mnt/lustre/work/gehler/gfh527/ni26_cov337_prepare_v2_20261117/teacher_fixture.npz')
FIXTURE_SHA='06644bae5a331022c755184ad71089b08c8e079271917e4a4223eafa8bd3d187'


def main():
    root=Path(os.environ['COV_ROOT']);api=Path(os.environ['REFERENCE336_ROOT']);start=time.monotonic();torch.set_num_threads(2)
    if sha(api/'source.sha256')!='7fa3da6d63270d9911de7d302c1f84e954ab02f0797818a6ae08ebca3b857383' or sha(api/'recipe336.py')!='468d632f51e52ffd16f1dff1ae61a2a4b22483a22a7618f25a6d36952d6df3be':raise ValueError('pinned336 API identity')
    guard=json.loads((api/'guard.json').read_text())
    if not guard['passed'] or not guard['native_observer_exact'] or guard['query_labels_opened']:raise ValueError('verified no-query336 observer guard required')
    if sha(api/'source.sha256')!=guard['source_sha256']:raise ValueError('336 guard/source identity')
    for line in (api/'source.sha256').read_text().splitlines():
        pin,name=line.split(maxsplit=1)
        if sha(api/name)!=pin:raise ValueError('336 source changed')
    sys.path.insert(0,str(api));import recipe336 as recipe
    d,allfit,query,people=recipe.load();fit=allfit[people[allfit]!='P09']
    if set(people[fit])!={'P04','P05','P06','P07','P08','P10'}:raise ValueError('source6 exclusion')
    y=recipe.labels(fit,len(people));d['y']=np.full(len(people),-1,np.int64);d['y'][fit]=y
    d.update(w=np.ones(len(people)),nolow=np.zeros(len(people),bool),tag='')
    if sha(FIXTURE)!=FIXTURE_SHA:raise ValueError('teacher fixture changed')
    with np.load(FIXTURE,allow_pickle=False) as z:
        rows=z['coverage3'][0];positions=np.searchsorted(z['rows'],rows)
        if not np.array_equal(z['rows'][positions],rows):raise ValueError('target map')
        target=torch.from_numpy(((z['targets'][positions]-z['mean'])/z['std']).astype(np.float32))
        exposure={name:{p:int(np.sum(people[z[name]]==p)) for p in sorted(set(people[z[name]].ravel()))} for name in ('coverage3','coverage6')}
        class_exposure={name:np.bincount(d['y'][z[name]].ravel(),minlength=3).tolist() for name in ('coverage3','coverage6')}
    if np.any(d['y'][rows]<0) or set(people[rows])-{'P04','P05','P06'}:raise ValueError('source pretext fixture')
    model=recipe.native(d);captured={}
    def capture(self,X,y,sw,r,epochs,lr,rng,gen,loss_fn,weighted,head_only=False,avg=None):
        if weighted or loss_fn.weight is not None or epochs!=40 or loss_fn.label_smoothing!=0:raise ValueError('native training recipe')
        self._prepare();captured.update(X=X,loss=type(loss_fn).__name__)
    model._train=types.MethodType(capture,model);model.fit(fit)
    order=np.argsort(model._order);pos=order[np.searchsorted(model._order[order],rows)]
    if not np.array_equal(model._order[pos],rows):raise ValueError('native preprocessing map')
    xb=captured['X'][pos];net=model.net;head=net.final_layer
    if not is_parametrized(head,'weight') or head.weight.shape!=(3,1152) or head.parametrizations.weight[0].max_norm!=.5:raise ValueError('original constrained classifier')
    native_head=head.parametrizations.weight.original;head_before=native_head.detach().clone();rng=torch.get_rng_state().clone()
    teacher=TeacherHead()
    if not torch.equal(rng,torch.get_rng_state()):raise ValueError('auxiliary initialization changed native RNG')
    tap={}
    def hook(module,args):tap['features']=args[0]
    handle=head.register_forward_pre_hook(hook);net.train()
    opt=torch.optim.AdamW(list(net.parameters())+list(teacher.parameters()),lr=.001,weight_decay=.001)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=8)
    body=[(n,p) for n,p in net.named_parameters() if p.requires_grad and 'final_layer' not in n]
    original=[p.detach().clone() for n,p in body];losses=[];gradients=[];tick=time.monotonic()
    try:
        for step in range(8):
            opt.zero_grad();net(xb);loss,groups=teacher(tap['features'],target);loss.backward()
            grads=[p.grad for n,p in body if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads) or native_head.grad is not None:raise ValueError('wrong gradient path')
            gradients.append(max(float(g.abs().max()) for g in grads));losses.append(groups.detach().tolist());opt.step();sched.step()
        with torch.no_grad():net(xb);final,final_groups=teacher(tap['features'],target)
    finally:handle.remove()
    seconds=time.monotonic()-tick;change=max(float((p.detach()-old).abs().max()) for (_,p),old in zip(body,original))
    if not torch.equal(native_head.detach(),head_before) or not is_parametrized(head,'weight'):raise ValueError('pretext altered native classifier')
    initial=float(np.mean(losses[0]));relative=(initial-float(final))/initial
    passed=relative>=.01 and change>0 and min(gradients)>0 and np.isfinite(losses).all()
    report=dict(passed=bool(passed),source_sha256=sha(root/'source.sha256'),reference336_guard_sha256=sha(api/'guard.json'),
        reference336_source_sha256=guard['source_sha256'],fixture_sha256=FIXTURE_SHA,source_people=sorted(set(people[fit])),
        native_scale=model.scale,native_preprocessed_batch_sha256=digest(xb.numpy()),batch_rows_sha256=digest(rows),
        class_counts=np.bincount(y,minlength=3).tolist(),exposure_person_counts=exposure,exposure_class_counts=class_exposure,
        group_losses_before_steps=losses,group_losses_after=final_groups.tolist(),relative_loss_decrease=relative,
        body_gradient_max=gradients,body_parameter_change_max=change,native_classifier_unchanged=True,auxiliary_RNG_preserved=True,
        eight_step_seconds=seconds,seconds=time.monotonic()-start,source_quality_score=False,P02P03_inference=False,
        boundary='optimization only; no adequate40epoch source6 quality control or scientific pretraining matrix released')
    (root/'optimization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not passed:raise ValueError('optimization guard failed; do not release scientific fit')


if __name__=='__main__':main()
