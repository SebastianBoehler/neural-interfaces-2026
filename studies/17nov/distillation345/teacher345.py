"""Classical task targets with exact inner-person prior/scaler/head exclusion."""
import json
import ctypes
import os
import time
from pathlib import Path
import numpy as np
from common345 import source,sha,digest,PRODUCER,PRODUCER_SHA,FIT,T,permutations,native_teacher


def caches(d,people):
    values={};hashes={}
    for p in (*FIT,'P09'):
        path=PRODUCER/f'runs/s301_native_inputs/{p}.npz';hashes[p]=sha(path)
        with np.load(path) as z:v={k:z[k].copy() for k in z.files}
        m=json.loads(str(v.pop('metadata')));r=v['rows']
        if m['person']!=p or m['source_sha256']!=PRODUCER_SHA or m['held_accuracy_computed'] or not np.array_equal(r,np.flatnonzero(people==p)):raise ValueError('personlocal cache provenance')
        # Original Python whole-session transform canary, using the cached covariances.
        s=d['stream'][r];group=np.flatnonzero(s==s[0]);i=group[0]
        expected=np.concatenate([T.whiten_tangent(v['cov'][i:i+1,b],T.expm(T.logm(v['cov'][group,b]).mean(0)))[0] for b in range(51)]+[(v['slow'][i]-v['slow'][group].mean(0)).astype(np.float64)])
        if not np.allclose(expected,v['features'][i],atol=2e-5,rtol=2e-5):raise ValueError('native cache versus original Python canary')
        values[p]=v
    return values,hashes


def python_causal(v,rows,prior):
    loc=np.searchsorted(v['rows'],rows);c=v['cov'][loc];s=v['slow'][loc].astype(np.float64);n=np.arange(1,len(rows)+1)
    out=[]
    for b in range(51):
        l=T.logm(c[:,b]);ref=T.expm((2*prior[b]+np.cumsum(l,0))/(2+n)[:,None,None])
        out.append(T.whiten_tangent(c[:,b],ref))
    out.append(s-np.cumsum(s,0)/(2+n[:,None]))
    result=np.concatenate(out,1)
    if not np.isfinite(result).all():raise ValueError('nonfinite original causal features')
    return result


def causal(v,rows,prior):
    loc=np.searchsorted(v['rows'],rows);lib=native_teacher()
    pd=ctypes.POINTER(ctypes.c_double);pf=ctypes.POINTER(ctypes.c_float)
    pp=np.ascontiguousarray(prior);cov=np.ascontiguousarray(v['cov'][loc]);slow=np.ascontiguousarray(v['slow'][loc]);result=np.empty((len(rows),44239),np.float64)
    handle=lib.core.nc_create(pp.ctypes.data_as(pd),0) # Explicit CPU, historical wrapper hardcodes CUDA1.
    if not handle:raise RuntimeError(lib.core.nc_error().decode())
    try:
        for i in range(len(rows)):
            lib.check(lib.core.nc_step(handle,cov[i].ctypes.data_as(pd),slow[i].ctypes.data_as(pf),result[i].ctypes.data_as(pd)),'nc_error')
    finally:lib.core.nc_destroy(handle)
    expected=python_causal(v,rows[:17],prior)
    if not np.allclose(result[:len(expected)],expected,atol=2e-5,rtol=2e-5):raise ValueError('native versus original Python source prefix')
    if not np.isfinite(result).all():raise ValueError('native feature nonfinite')
    return result


def fit_predict(target,values,d,people,y,perms,smoke=False):
    fitting=[p for p in FIT if p!=target];rows=np.sort(np.concatenate([values[p]['rows'] for p in fitting]))
    if set(people[rows])!=set(fitting) or target in set(people[rows]) or 'P09' in set(people[rows]):raise ValueError('strict nested aggregate exclusion')
    f=np.concatenate([values[p]['features'] for p in fitting]);rr=np.concatenate([values[p]['rows'] for p in fitting]);order=np.argsort(rr);f=f[order]
    yy=d['y'][rows]
    if (yy<0).any() or not np.array_equal(rows,rr[order]):raise ValueError('source labels/rows')
    tick=time.monotonic();model=T.fit_head('lr0.3',f,yy,[])
    prior=np.average(np.stack([values[p]['prior'] for p in fitting]),axis=0,weights=[len(values[p]['rows']) for p in fitting])
    probabilities={};query=values[target]['rows'];before=digest(prior)
    orders=('within0',) if target!='P09' or smoke else tuple(perms)
    for name in orders:
        delivered=perms[name][people[perms[name]]==target];out=np.empty((len(query),3))
        for session in dict.fromkeys(d['stream'][delivered]):
            r=delivered[d['stream'][delivered]==session];use=r[:17] if smoke else r
            feat=causal(values[target],use,prior)
            if not np.array_equal(feat[:min(7,len(use))],causal(values[target],use[:7],prior)):raise ValueError('prefix future dependence')
            pp=model.predict_proba(feat)
            if smoke:out=np.empty_like(pp);out[:]=pp;break
            out[np.searchsorted(query,use)]=pp
        probabilities[name]=out
    if digest(prior)!=before:raise ValueError('prior mutated across sessions')
    record=dict(target=target,fit_people=fitting,fit_rows_sha256=digest(rows),prior_sha256=before,
        scaler_mean_sha256=digest(model.steps[0][1].mean_),scaler_scale_sha256=digest(model.steps[0][1].scale_),
        coefficients_sha256=digest(model.steps[1][1].coef_),iterations=model.steps[1][1].n_iter_.tolist(),seconds=time.monotonic()-tick)
    if max(record['iterations'])>=2000:raise ValueError('original LR did not converge')
    return probabilities,record


def main():
    root=Path(os.environ['DISTILL_ROOT']);d,fit,held,people,y=source();perms=permutations(len(people));v,hashes=caches(d,people)
    preds={};records={};start=time.monotonic()
    for p in (*FIT,'P09'):
        preds[p],records[p]=fit_predict(p,v,d,people,y,perms)
        print(p,records[p]['seconds'],flush=True)
    oof=np.empty((len(fit),3))
    for p in FIT:oof[np.searchsorted(fit,v[p]['rows'])]=preds[p]['within0']
    order=np.arange(len(fit));rng=np.random.default_rng(345)
    for p in FIT:
        for c in range(3):
            r=np.flatnonzero((people[fit]==p)&(y==c));r=r[rng.permutation(len(r))]
            if len(r)<2:raise ValueError('cannot derange person/class')
            order[r]=np.roll(r,1)
    if (order==np.arange(len(fit))).any() or not np.array_equal(np.sort(order),np.arange(len(fit))):raise ValueError('posterior derangement')
    out=root/'teacher.npz'
    with out.open('xb') as f:np.savez(f,index=fit,proba=oof,deranged=oof[order],derangement=order,query=held,**preds['P09'])
    report=dict(teacher_sha256=sha(out),source_sha256=sha(root/'source.sha256'),cache_hashes=hashes,records=records,
        source_rows_sha256=digest(fit),held_rows_sha256=digest(held),source_BA=float(np.mean([np.mean(oof[y==c].argmax(1)==c) for c in range(3)])),
        seconds=time.monotonic()-start,held_labels_opened=False,scope='core-only source-person-OOF;not fullsystem')
    (root/'teacher.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
