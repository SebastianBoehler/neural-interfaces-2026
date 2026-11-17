"""Interleaved late-versus-repeated official supervision on the original native model."""
import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.parametrize import is_parametrized

DEPENDENCY=Path('/mnt/lustre/work/gehler/gfh527/ni26_direct346_v1_20261117')
sys.path.insert(0,str(DEPENDENCY))
from common346 import source,native,sha,digest,state_sha,CONTROL
PREPARATION=Path('/mnt/lustre/work/gehler/gfh527/ni26_crops347_prepare_v2_20261117')


def late_data(fit):
    card=json.loads((PREPARATION/'qualification.json').read_text())
    if sha(PREPARATION/'source.sha256')!='969b88ad1e092a0e412d6ef2cc914b1c1d75608e7f3d6bdcb04f6037e45c40b1' or card['source_sha256']!=sha(PREPARATION/'source.sha256'):raise ValueError('pinned qualification source changed')
    for line in (PREPARATION/'source.sha256').read_text().splitlines():
        pin,name=line.split(maxsplit=1)
        if sha(PREPARATION/name)!=pin:raise ValueError('qualified preprocessing changed:'+name)
    if not card['passed'] or card['held_labels_opened'] or card['query_transformed'] or card['original_source_rows_sha256']!=digest(fit):raise ValueError('passed strict source qualification required')
    if sha(PREPARATION/'late.npz')!=card['late_sha256'] or card['late_sha256']!='c1dcb7979a53525ac5edbff81dcdbe01c9fdc2d40ec166b182ed48da97cd6042':raise ValueError('qualified late input changed')
    with np.load(PREPARATION/'late.npz') as z:rows=z['index'].copy();x=z['X'].copy()
    if len(fit)!=1898 or len(rows)!=1894 or np.setdiff1d(rows,fit).size or x.shape!=(1894,41,480):raise ValueError('exact eligible subset/shape differs')
    if digest(rows)!=card['summary']['extra_rows_sha256'] or digest(x)!=card['late_input_sha256']:raise ValueError('qualified source row/input identity')
    return rows,x,card


def attach(model,arm,eligible,late,reference):
    if arm not in ('A','B','C'):raise ValueError('unregistered crop arm')
    original_train=model._train;trace=dict(arm=arm,steps=0,base_batches=[],extra_batches=[],
        step_seconds=[],body_gradient_max=0.,head_change_max=0.,head_effective_norm_max=0.)
    def train(self,X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only=False,avg=None):
        if self.align or self.crops or self.aug or self.rot or self.dev!='cpu' or weighted or head_only or avg is not None:raise ValueError('changed native source route')
        if any(getattr(self,k) for k in ('am','coral','dann','dist','swad','cotrain','adabn')):raise ValueError('unregistered auxiliary policy')
        if loss_fn.weight is not None or loss_fn.label_smoothing or epochs not in (1,40):raise ValueError('native criterion/epochs changed')
        trace['initial_state_sha256']=state_sha(self.net)
        if trace['initial_state_sha256']!=reference['trace']['initial_state_sha256'] or self.scale!=reference['trace']['scale']:raise ValueError('originalA6 initializer/source scale identity')
        self._prepare();head=self.net.final_layer
        if not is_parametrized(head,'weight') or head.weight.shape!=(3,1152) or head.parametrizations.weight[0].max_norm!=.5:raise ValueError('native constrained head changed')
        n=len(X);base=np.arange(n);position=np.flatnonzero(np.isin(self._order,eligible));lookup=np.full(n,-1,int);lookup[position]=np.arange(len(position))
        latepos=np.searchsorted(eligible,self._order[position])
        if not np.array_equal(eligible[latepos],self._order[position]) or len(position)!=1894:raise ValueError('extra eligible native row map')
        extra=X[position].clone() if arm!='C' else torch.from_numpy((late[latepos].astype(np.float64)/self.scale).astype(np.float32))
        if arm=='C' and torch.equal(extra,X[position]):raise ValueError('late EEG route inactive')
        trace.update(source_order_sha256=digest(self._order),original_input_sha256=digest(X.numpy()),
            extra_rows_sha256=digest(self._order[position]),extra_input_sha256=digest(extra.numpy()),
            original_labels_sha256=digest(y.numpy()),extra_labels_sha256=digest(y[position].numpy()),
            extra_is_original_copy=bool(torch.equal(extra,X[position])),scale=self.scale,epochs=epochs,
            torch_rng_before=digest(torch.get_rng_state().numpy()),augment_rng_before=digest(gen.get_state().numpy()))
        old_batches=self._batches;extra_rng=np.random.default_rng(347)
        def batches(_,supplied,generator):
            bb=old_batches(base,generator)
            if arm=='A':
                if generator is rng:trace['base_batches'].append(digest(np.concatenate([self._order[b] for b in bb])))
                return bb
            ee=old_batches(position,extra_rng if generator is rng else np.random.default_rng(347))
            if len(bb)!=31 or len(ee)!=31:raise ValueError('native base/eligible batchcounts not exact31; no fallback')
            for group in ee:
                for chunk in group.reshape(-1,32):
                    if len(set(self.d['stream'][self._order[chunk]]))!=1 or (lookup[chunk]<0).any():raise ValueError('extra same-session/eligibility guard')
            if generator is rng:
                trace['base_batches'].append(digest(np.concatenate([self._order[b] for b in bb])))
                trace['extra_batches'].append(digest(np.concatenate([self._order[b] for b in ee])))
            return [b for pair in zip(bb,[n+lookup[e] for e in ee]) for b in pair]
        self._batches=types.MethodType(batches,self)
        train_x=X if arm=='A' else torch.cat([X,extra]);train_y=y if arm=='A' else torch.cat([y,y[position]])
        train_sw=sw if arm=='A' or sw is None else torch.cat([sw,sw[position]])
        clock={};old_step=torch.optim.AdamW.step
        def forward_start(module,args):clock['start']=time.monotonic()
        handle=self.net.register_forward_pre_hook(forward_start)
        def step(optimizer,*args,**kwargs):
            param=head.parametrizations.weight.original;parameters=[p for g in optimizer.param_groups for p in g['params']]
            if sum(p is param for p in parameters)!=1 or param.grad is None or float(param.grad.abs().max())<=0:raise ValueError('head gradient/optimizer no-op')
            body=[p.grad for name,p in self.net.named_parameters() if 'final_layer' not in name and p.grad is not None]
            if not body or not all(torch.isfinite(g).all() for g in body):raise ValueError('invalid native body gradient')
            trace['body_gradient_max']=max(trace['body_gradient_max'],max(float(g.abs().max()) for g in body))
            before=param.detach().clone();value=old_step(optimizer,*args,**kwargs);change=float((param.detach()-before).abs().max())
            if change<=0:raise ValueError('head parameter update no-op')
            trace['head_change_max']=max(trace['head_change_max'],change);trace['steps']+=1
            trace['step_seconds'].append(time.monotonic()-clock['start'])
            trace['head_effective_norm_max']=max(trace['head_effective_norm_max'],float(head.weight.detach().norm(dim=1).max()))
            return value
        torch.optim.AdamW.step=step
        try:original_train(train_x,train_y,train_sw,np.arange(len(train_x)),epochs,lr,rng,gen,loss_fn,weighted,head_only,avg)
        finally:torch.optim.AdamW.step=old_step;handle.remove();self._batches=old_batches
        expected=epochs*31*(1 if arm=='A' else 2)
        if trace['steps']!=expected or trace['body_gradient_max']<=0 or trace['head_effective_norm_max']>.500001:raise ValueError('native step/activity/constraint guard failed')
        trace.update(final_state_sha256=state_sha(self.net),torch_rng_after=digest(torch.get_rng_state().numpy()),augment_rng_after=digest(gen.get_state().numpy()))
    model._train=types.MethodType(train,model)
    return trace


def run(arm,epochs=40):
    d,fit,held,people,y=source();eligible,late,card=late_data(fit);reference=json.loads((CONTROL/'record.json').read_text())
    model=native(d,epochs=epochs);trace=attach(model,arm,eligible,late,reference);tick=time.monotonic();model.fit(fit);model.set_k(20.)
    return d,fit,held,people,y,model,trace,card,time.monotonic()-tick
