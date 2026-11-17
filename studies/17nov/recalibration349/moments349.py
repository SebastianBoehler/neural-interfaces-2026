"""Actual pre-BN activations, native source-chunk EMA and independent moment oracles."""
import copy
import numpy as np
import torch
from source349 import bn,digest,state_sha


@torch.no_grad()
def phase(model,d,rows,raw,perm,fixture=False):
    if any(m.training for m in model.net.modules()) or raw.shape!=(len(rows),41,480):raise ValueError('frozen current-window input route')
    first=np.empty((len(rows),288),np.float32);second=np.empty_like(first);feature=np.empty((len(rows),1152),np.float32);proba=np.empty((len(rows),3));saved={}
    rank=np.empty(len(perm),int);rank[perm]=np.arange(len(perm));state=state_sha(model.net)
    tap={}
    def pre(module,args):
        x=args[0];red=tuple(range(2,x.ndim));tap['r1']=x.mean(red).numpy().copy();tap['r2']=x.square().mean(red).numpy().copy()
        if fixture and 'activation' not in saved:saved['activation']=x[:64].detach().clone()
    def head(module,args):tap['features']=args[0].numpy().copy()
    a=bn(model).register_forward_pre_hook(pre);b=model.net.final_layer.register_forward_pre_hook(head)
    try:
        for stream in dict.fromkeys(d['stream'][rows]):
            loc=np.flatnonzero(d['stream'][rows]==stream);loc=loc[np.argsort(rank[rows[loc]],kind='stable')]
            x=torch.from_numpy((raw[loc].astype(np.float64)/model.scale).astype(np.float32))
            proba[loc]=torch.softmax(model.net(x).double(),1).numpy();first[loc]=tap['r1'];second[loc]=tap['r2'];feature[loc]=tap['features']
    finally:a.remove();b.remove()
    if state_sha(model.net)!=state or not all(np.isfinite(v).all() for v in (first,second,feature,proba)):raise ValueError('source extraction mutated state/nonfinite')
    return dict(first=first,second=second,feature=feature,proba=proba),saved


def chunk_moments(first,second):
    first=first.reshape(-1,32,288);second=second.reshape(-1,32,288)
    gm=first.mean(1);gv=(second.mean(1)-gm.square()).clamp_min(0)
    return gm.mean(0),gv.mean(0)


@torch.no_grad()
def oracle(model,activation):
    if activation.shape!=(64,288,1,480):raise ValueError('real source oracle requires64×288×1×480')
    original=bn(model);child=copy.deepcopy(original);child.train()
    first=activation.mean((2,3));second=activation.square().mean((2,3));mu,var=chunk_moments(first,second)
    expected_mu=.9*child.running_mean+.1*mu;expected_var=.9*child.running_var+.1*var
    child(activation)
    error=max(float((child.running_mean-expected_mu).abs().max()),float((child.running_var-expected_var).abs().max()))
    independent=activation.reshape(2,32,288,1,480).permute(0,2,1,3,4).reshape(2,288,-1).var(-1,correction=0).mean(0)
    variance_error=float((var-independent).abs().max())
    # Native eval/current-prefix second-moment consumer, with real new source priors.
    child.eval();x=activation[:7];r1=x.mean((2,3));r2=x.square().mean((2,3));n=torch.arange(1,8)[:,None]
    mean=(20*child.running_mean+r1.cumsum(0))/(20+n)
    vv=((20*(child.running_var+child.running_mean.square())+r2.cumsum(0))/(20+n)-mean.square()).clamp_min(0)
    expected=(x-mean[:,:,None,None])/torch.sqrt(vv[:,:,None,None]+child.eps)
    expected=expected*child.weight[None,:,None,None]+child.bias[None,:,None,None]
    consumer_error=float((child(x)-expected).abs().max())
    if error>1e-6 or variance_error>1e-6 or consumer_error>1e-6:raise ValueError('actual biasedchunk/EMA/eval consumer oracle failed')
    return dict(native_update_maxabs=error,independent_biasedvar_maxabs=variance_error,native_prefix_consumer_maxabs=consumer_error)


@torch.no_grad()
def recalibrate(model,fit,eligible,official,late,expected,mixed):
    if digest(model._order)!=expected['trace']['source_order_sha256']:raise ValueError('source order differs')
    mean=torch.zeros(288);var=torch.ones(288);rng=np.random.default_rng(0);extra_rng=np.random.default_rng(347);hashes=[];extra_hashes=[];updates=0
    eligible_positions=np.flatnonzero(np.isin(model._order,eligible))
    for _ in range(40):
        batches=model._batches(np.arange(len(fit)),rng);hashes.append(digest(np.concatenate([model._order[b] for b in batches])))
        extras=model._batches(eligible_positions,extra_rng);extra_hashes.append(digest(np.concatenate([model._order[b] for b in extras])))
        if len(batches)!=31 or len(extras)!=31:raise ValueError('actual original31+eligible31batch recipe differs')
        for base,extra in zip(batches,extras):
            for batch,use_late in ((base,False),(extra,mixed)):
                rows=model._order[batch];rr=eligible if use_late else fit;values=late if use_late else official;pos=np.searchsorted(rr,rows)
                if not np.array_equal(rr[pos],rows):raise ValueError('source moment row map')
                first=torch.from_numpy(values['first'][pos]);second=torch.from_numpy(values['second'][pos]);mu,v=chunk_moments(first,second)
                mean.mul_(.9).add_(.1*mu);var.mul_(.9).add_(.1*v);updates+=1
    if hashes!=expected['trace']['base_batches'] or extra_hashes!=expected['trace']['extra_batches'] or updates!=2480:raise ValueError('ALL40 original+extra source batchhash identity failed')
    return mean.numpy(),var.numpy(),dict(updates=updates,base_batch_hashes=hashes,extra_batch_hashes=extra_hashes,source_rows_sha256=digest(fit),extra_rows_sha256=digest(eligible),extra_phase='late' if mixed else 'official_repeat')


def quality(proba,y):
    return dict(pooledNLL=float(-np.log(np.maximum(proba[np.arange(len(y)),y],1e-12)).mean()),class_recalls=[float(np.mean(proba[y==c].argmax(1)==c)) for c in range(3)])


def geometry(original,late,y):
    std=original.std(0)
    if np.any(std<=1e-12):raise ValueError('source class geometry has constant coordinate')
    a=np.stack([original[y==c].mean(0) for c in range(3)])/std;b=np.stack([late[y==c].mean(0) for c in range(3)])/std
    result={}
    for i,j in ((0,1),(0,2),(1,2)):
        x=a[i]-a[j];z=b[i]-b[j];nx=np.linalg.norm(x);nz=np.linalg.norm(z)
        if min(nx,nz)<=1e-12:raise ValueError('degenerate source class direction')
        result[f'{i}-{j}']=dict(phase_direction_cosine=float(np.dot(x,z)/(nx*nz)),late_original_length_ratio=float(nz/nx))
    return result


def describe(original,late,positions,y):
    a=original['first'][positions].astype(np.float64);b=late['first'].astype(np.float64)
    av=np.maximum(original['second'][positions]-a*a,1e-12);bv=np.maximum(late['second']-b*b,1e-12)
    va=original['second'][positions].mean(0)-a.mean(0)**2;vb=late['second'].mean(0)-b.mean(0)**2
    if np.any(va<=0) or np.any(vb<=0):raise ValueError('nonpositive global phase variance')
    return dict(preBN_mean_shift_RMS=float(np.sqrt(np.mean((b.mean(0)-a.mean(0))**2/va))),
        preBN_global_logvariance_shift_RMS=float(np.sqrt(np.mean(np.log(vb/va)**2))),
        preBN_moment_class_geometry=geometry(np.c_[a,np.log(av)],np.c_[b,np.log(bv)],y),
        penultimate_class_geometry=geometry(original['feature'][positions].astype(np.float64),late['feature'].astype(np.float64),y))
