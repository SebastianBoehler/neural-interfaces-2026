"""Add an active auxiliary to the actual native loop, without replacing the loop."""
import copy
import time
import types
import numpy as np
import torch
from common344 import digest,state_sha,GROUPS,embedding


def derange(rows,people,labels,random):
    result=rows.copy()
    for person in np.unique(people[rows]):
        for cls in np.unique(labels[rows][people[rows]==person]):
            where=np.flatnonzero((people[rows]==person)&(labels[rows]==cls));values=rows[where]
            for _ in range(128):
                candidate=values[random.permutation(len(values))]
                if np.all(candidate!=values):break
            else:raise ValueError('within-person/class derangement impossible')
            result[where]=candidate
            if not np.array_equal(np.sort(candidate),np.sort(values)):raise ValueError('target multiset')
    if np.any(result==rows) or not np.array_equal(people[result],people[rows]) or not np.array_equal(labels[result],labels[rows]):raise ValueError('derangement grouping')
    return result


def snapshot(model,aux,folder,name,fit,held):
    mode=model.net.training;ks=[m.k for m in model.sbns()]
    features=embedding(model,np.r_[fit,held])
    for m,k in zip(model.sbns(),ks):m.k=k
    model.net.train(mode)
    path=folder/f'{name}.pt';torch.save(dict(body=model.net.state_dict(),aux=None if aux is None else aux.state_dict(),G=model.G,scale=model.scale),path)
    np.savez(folder/f'{name}.npz',rows=np.r_[fit,held],features=features)


def continue_fit(model,aux,supervised,X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only=False,avg=None,*,arm,teacher,fit,held,people,folder,record,limit=None,save=True):
    if save:snapshot(model,aux,folder,'fine0',fit,held)
    active=arm in ('C','D');zero=arm=='Z';body=[p for n,p in model.net.named_parameters() if p.requires_grad and 'final_layer' not in n]
    if active:
        if aux is None or aux.out_features!=44239:raise ValueError('auxiliary dimensions')
        model.net.add_module('retention_aux',aux)
    position=np.full(len(model.d['X']),-1,int);position[teacher['rows']]=np.arange(len(teacher['rows']))
    old_batches=model._batches;batchinfo={};epoch=[0];counter=[0];random=np.random.default_rng(344);tap={};alpha=[None]
    record.update(aux_losses=[],task_losses=[],derangement_hashes=[],batch_hashes=[],alpha=None,steps=0)
    def batches(self,r,generator):
        values=old_batches(r,generator)
        if generator is rng:
            if epoch[0]==10 and save:snapshot(model,aux,folder,'fine10',fit,held)
            flat=np.concatenate([self._order[j] for j in values]);record['batch_hashes'].append(digest(flat));target=derange(flat,people,self.d['y'],random) if arm=='D' else flat
            if arm=='D':record['derangement_hashes'].append(digest(target))
            batchinfo['batches']=values;batchinfo['targets']=np.split(target,np.cumsum([len(j) for j in values])[:-1]);counter[0]=0;epoch[0]+=1
        return values
    model._batches=types.MethodType(batches,model)
    def features(module,args):tap['features']=args[0]
    def loss_hook(module,args,value):
        if not(active or zero):return value
        j=counter[0];targetrows=batchinfo['targets'][j];counter[0]+=1
        if not np.isin(targetrows,fit).all():raise ValueError('aux target outside source')
        if zero:return value
        target=torch.from_numpy(teacher['covariance'][position[targetrows]])
        prediction=aux(tap['features']);error=(prediction-target).square();auxloss=torch.stack([error[:,g].mean() for g in GROUPS]).mean()
        if alpha[0] is None:
            true_target=torch.from_numpy(teacher['covariance'][position[model._order[batchinfo['batches'][j]]]])
            calibration_error=(prediction-true_target).square();calibration_loss=torch.stack([calibration_error[:,g].mean() for g in GROUPS]).mean()
            a=torch.autograd.grad(value,body,retain_graph=True,allow_unused=True);b=torch.autograd.grad(calibration_loss,body,retain_graph=True,allow_unused=True)
            norm=lambda gs:torch.stack([g.detach().square().sum() for g in gs if g is not None]).sum().sqrt()
            na,nb=float(norm(a)),float(norm(b))
            if not np.isfinite([na,nb]).all() or min(na,nb)<=0:raise ValueError('source gradient normalization')
            alpha[0]=.1*na/nb;record.update(alpha=alpha[0],task_gradient_norm=na,aux_gradient_norm=nb)
        record['task_losses'].append(float(value.detach()));record['aux_losses'].append(float(auxloss.detach()));record['steps']+=1
        return value+alpha[0]*auxloss
    h=model.net.final_layer.register_forward_pre_hook(features);l=loss_fn.register_forward_hook(loss_hook)
    old_step=torch.optim.AdamW.step;steps=[0]
    def step(opt,*a,**kw):
        value=old_step(opt,*a,**kw);steps[0]+=1
        if active and (aux.weight.grad is None or not torch.isfinite(aux.weight.grad).all() or float(aux.weight.grad.abs().max())<=0):raise ValueError('aux optimizer inactive')
        if limit and steps[0]>=limit:raise StopIteration('source smoke limit')
        return value
    torch.optim.AdamW.step=step;start=time.monotonic()
    try:supervised(X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only,avg)
    except StopIteration:
        if limit is None:raise
    finally:
        torch.optim.AdamW.step=old_step;h.remove();l.remove();model._batches=old_batches
        if active:delattr(model.net,'retention_aux')
    record.update(fine_seconds=time.monotonic()-start,actual_steps=steps[0],epochs_started=epoch[0],torch_rng_after=digest(torch.get_rng_state().numpy()))
    if active and record['steps']!=steps[0]:raise ValueError('aux objective not applied every step')
    if save:snapshot(model,aux,folder,'fine40',fit,held)
