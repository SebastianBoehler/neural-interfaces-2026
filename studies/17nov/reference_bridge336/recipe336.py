"""Exact native reference; adapter observes or changes exactly one registered factor."""
import hashlib
import inspect
import json
import sys
import textwrap
import types
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized,remove_parametrizations

PARENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_oscillatory331_v2_20261117')
sys.path.insert(0,str(PARENT))
from data331 import BASE,load,labels,sha,digest,official
sys.path.insert(0,str(BASE/'experiments'))
import deep,session_bn


def state_sha(model):
    h=hashlib.sha256()
    for n,p in sorted(model.state_dict().items()):h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def source():
    if sha(PARENT/'source.sha256')!='b0fec93e6c3c6e3f66cc1d967e2c3b7ba1c87e01ca8308d3d56a6f5e0fe8fde9':raise ValueError('sourceAPIchanged')
    d,fit,query,people=load();y=labels(fit,len(d['X']));d['y']=np.full(len(d['X']),-1,np.int64);d['y'][fit]=y
    d.update(w=np.ones(len(d['X'])),nolow=np.zeros(len(d['X']),bool),tag='')
    for mod,name in ((deep,'deep.py'),(session_bn,'session_bn.py')):
        if Path(mod.__file__).resolve()!=(BASE/'experiments'/name).resolve():raise ValueError('wrongnativeimport')
    return d,fit,query,people,y


def native(d,epochs=40):
    model=session_bn.make('fbcnet_raw_sbn',d,0)
    expected={'arch':'fbcnet','align':False,'seed':0,'epochs':40,'lr':.001,'wd':.001,'batch':64,'group':32,'aug':False,'rot':0.}
    for name,value in expected.items():
        if getattr(model,name)!=value:raise ValueError(f'nativepresetdiffers:{name}')
    if model.dev!='cpu':raise ValueError('registeredCPUrouteonly')
    model.epochs=epochs
    return model


def prototype_batches(d,fit,order,epochs):
    rng=np.random.default_rng(0);position=np.full(len(d['X']),-1,int);position[order]=np.arange(len(order));out=[]
    for _ in range(epochs):
        chunks=[]
        for stream in np.unique(d['stream'][fit]):
            r=fit[d['stream'][fit]==stream];r=r[rng.permutation(len(r))];pad=(-len(r))%32
            if pad:r=np.r_[r,rng.choice(r,pad,replace=pad>len(r))]
            chunks.extend(list(r.reshape(-1,32)))
        chunks=[chunks[i] for i in rng.permutation(len(chunks))]
        out.append([position[np.concatenate(chunks[i:i+2])] for i in range(0,len(chunks),2)])
    if any((b<0).any() for ep in out for b in ep):raise ValueError('batchoutsidefit')
    return out


def scalar_patch():
    text=textwrap.dedent(inspect.getsource(deep.AlignedNet._fit))
    anchor='self.scale = float(np.sqrt(np.trace(self.G) / len(self.G)))'
    if text.count(anchor)!=1:raise ValueError('scaleanchorchanged')
    changed=text.replace(anchor,"self.scale = float(torch.as_tensor(self.d['X'][idx]).float().square().mean().sqrt())")
    env=dict(vars(deep));exec(compile(changed,'336_source_RMS_only','exec'),env)
    deep.AlignedNet._fit=env['_fit']


def observe(model,arm,fit):
    if arm not in ('A','H','U','W','B'):raise ValueError('unregisteredbridgearm')
    if arm=='U':scalar_patch()
    trace={'arm':arm,'batches':[],'steps':0,'max_gradient':0.,'head_effective_norm_max':0.}
    original_train=model._train
    def train(self,X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only=False,avg=None):
        head=self.net.final_layer
        if not isinstance(head,nn.Linear) or head.weight.shape!=(3,1152) or not is_parametrized(head,'weight'):
            raise ValueError('native1152constrainedheadnotpreserved')
        if len(head.parametrizations.weight)!=1 or head.parametrizations.weight[0].max_norm!=.5:raise ValueError('nativeconstraintdiffers')
        trace['initial_state_sha256']=state_sha(self.net)
        trace['head_class']=type(head).__name__;trace['head_code_sha256']=sha(inspect.getfile(type(head).__mro__[1]))
        trace['raw_initial_head_norm']=float(head.parametrizations.weight.original.detach().norm(dim=1).max())
        trace['effective_initial_head_norm']=float(head.weight.detach().norm(dim=1).max())
        before=torch.get_rng_state().clone()
        if arm=='H':
            parameter=head.parametrizations.weight.original
            remove_parametrizations(head,'weight',leave_parametrized=False)
            if head.weight is not parameter or is_parametrized(head,'weight') or not torch.equal(before,torch.get_rng_state()):raise ValueError('headtogglechangedtensor/RNG')
        counts=torch.bincount(y,minlength=3).float();weights=len(y)/(3*counts)
        trace['class_counts']=counts.tolist();trace['class_weights']=weights.tolist()
        if arm=='W':loss_fn=nn.CrossEntropyLoss(weight=weights,label_smoothing=loss_fn.label_smoothing)
        trace['criterion_weights']=None if loss_fn.weight is None else loss_fn.weight.tolist()
        trace['hyper']={k:getattr(self,k) for k in ('epochs','lr','wd','batch','group','aug','rot','align','seed','dev')}
        trace['scale']=self.scale;trace['source_order_sha256']=digest(self._order)
        trace['torch_rng_before']=digest(torch.get_rng_state().numpy());trace['augment_rng_before']=digest(gen.get_state().numpy())
        alternative=prototype_batches(self.d,fit,self._order,epochs);old_batches=self._batches;counter=0
        def batches(_,r,generator):
            nonlocal counter
            actual=generator is rng
            result=alternative[counter if actual else 0] if arm=='B' else old_batches(r,generator)
            if actual:
                trace['batches'].append(digest(np.concatenate([self._order[b] for b in result])));counter+=1
            return result
        self._batches=types.MethodType(batches,self)
        trace['full_step_seconds']=[];clock={}
        def forward_start(module,args):clock['start']=time.monotonic()
        begin_hook=self.net.register_forward_pre_hook(forward_start)
        old_step=torch.optim.AdamW.step
        def step(optimizer,*args,**kwargs):
            head_parameter=head.parametrizations.weight.original if is_parametrized(head,'weight') else head.weight
            parameters=[p for group in optimizer.param_groups for p in group['params']]
            if sum(p is head_parameter for p in parameters)!=1 or not head_parameter.requires_grad or head_parameter.grad is None or float(head_parameter.grad.abs().max())<=0:raise ValueError('actualheadnotinoptimizer/active')
            head_before=head_parameter.detach().clone()
            grads=[p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('nativegradientinvalid')
            trace['max_gradient']=max(trace['max_gradient'],max(float(g.abs().max()) for g in grads));trace['steps']+=1
            value=old_step(optimizer,*args,**kwargs)
            change=float((head_parameter.detach()-head_before).abs().max())
            if change<=0:raise ValueError('actualheadoptimizersilentno-op')
            trace['head_update_max']=max(trace.get('head_update_max',0.),change)
            trace['optimizer_owns_actual_head']=True
            trace['full_step_seconds'].append(time.monotonic()-clock['start'])
            return value
        def head_hook(module,args,out):
            trace['head_effective_norm_max']=max(trace['head_effective_norm_max'],float(module.weight.detach().norm(dim=1).max()))
        hook=head.register_forward_hook(head_hook);torch.optim.AdamW.step=step
        try:original_train(X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only,avg)
        finally:torch.optim.AdamW.step=old_step;hook.remove();begin_hook.remove();self._batches=old_batches
        if counter!=epochs or trace['max_gradient']<=0:raise ValueError('observednativefitno-op')
        trace['torch_rng_after']=digest(torch.get_rng_state().numpy());trace['augment_rng_after']=digest(gen.get_state().numpy())
        trace['final_state_sha256']=state_sha(self.net);trace['head_is_parametrized']=is_parametrized(head,'weight')
    model._train=types.MethodType(train,model)
    return trace


def inference(model,d,fit,query,yfit):
    reference=PARENT/'reference331.py'
    if sha(reference)!='b655935aa8fea6fdc35fc662b007def14d4e3342d2c705090ecc7cb22e9e703b':raise ValueError('referenceinferenceAPIchanged')
    text=reference.read_text();make="model = session_bn.make('fbcnet_raw_sbn', data, 0)";training='model.fit(fit); model.set_k(20.)'
    if text.count(make)!=1 or text.count(training)!=1:raise ValueError('referenceadapteranchorschanged')
    text=text.replace(make,'model = supplied_model',1).replace(training,'model.set_k(20.)',1)
    env={'__name__':'same_reference_inference','__file__':str(reference),'supplied_model':model}
    exec(compile(text,str(reference),'exec'),env)
    return env['predict'](d,fit,query,yfit)
