"""One guarded A100 source-only learned-display matrix, no outer continuation."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from support325 import setup,build,features,labels,digest,sha,state_sha,fixed_features,reference
from renderer325 import Renderer
from checks325 import run
from learn325 import extract,train,encode


def direct_guard(renderer,x):
    with torch.no_grad():
        a=renderer.direct(x[:1]); b=renderer.direct(x)
        if not torch.equal(a,renderer.direct(x[:1])):raise ValueError('directreset failed')
        if float((a-b[:1]).abs().max())>1e-6:raise ValueError('direct batch prefix failed')
        amp=float((renderer.direct(x[:1]*2)-a).abs().max())
        tail=x[:1].clone();tail[:,0,-1]+=renderer.scale
        last=float((renderer.direct(tail)-a).abs().max())
        if min(amp,last)<=1e-7:raise ValueError('direct amplitude/finalsample inactive')
    loss=renderer.direct(x).square().mean()
    grads=torch.autograd.grad(loss,list(renderer.parameters()))
    if any(not torch.isfinite(g).all() or float(g.abs().max())<=1e-12 for g in grads):raise ValueError('directgradient inactive')
    return dict(amplitude=amp,finalsample=last)


def main():
    root=Path(os.environ['VISUAL_ROOT']); started=time.monotonic()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():raise ValueError('registered A100required')
    guard=json.loads((root/'guard.json').read_text())
    if not guard['passed'] or guard['labels_opened'] or guard['source_sha256']!=sha(root/'source.sha256') or guard['feature_sha256']!=sha(root/'guard_features.npz'):
        raise ValueError('passed source-locked no-score guard required')
    d,fit,val,people,base,meta=setup(root); rows=np.r_[fit,val];near=np.array(guard['neighbors'])
    probe=torch.from_numpy(d['X'][guard['source_rows']].copy()).cuda()
    engineering={}; projected=300.
    for variant in ('pretrained','random'):
        renderer=Renderer(base,near).cuda();encoder=build(root,variant,'cuda')
        engineering[variant]=run(renderer,encoder,probe,'cuda')
        with torch.no_grad(),np.load(root/'guard_features.npz') as z:
            error=float(np.max(np.abs(features(encoder,renderer(probe[:1]),1435).cpu().numpy()-z[variant])))
        if error>.005:raise ValueError('CPU/A100 feature parity failed')
        engineering[variant]['CPU_A100_max_abs']=error
        timing=engineering[variant]['timing']
        projected+=3*len(fit)/4*timing['renderer_backward_B4_seconds']+len(rows)/4*timing['forward_B4_seconds']
        del renderer,encoder
    if projected>1440:raise RuntimeError(f'no-label measured projection {projected:.1f}s exceeds1440budget')
    print(json.dumps(dict(no_score_A100_passed=True,projected_seconds=projected,engineering=engineering)),flush=True)
    # True labels are restricted to seven fitpeople and opened after BOTH engineering guards.
    yfit=labels(fit,len(d['X'])); banks={}; training={}; final_checks={}; checkpoints={}
    raw,_=fixed_features(rows,'pretrained'); banks['raw287']=raw
    for variant in ('pretrained','random','direct'):
        encoder=None if variant=='direct' else build(root,variant,'cuda')
        fixed=Renderer(base,near).cuda()
        initial=extract(fixed,encoder,d['X'][rows])
        if encoder is not None:
            _,original=fixed_features(rows,variant)
            if float(np.max(np.abs(initial-original)))>.005:raise ValueError('fixed variant differs from324')
        for learned in (False,True):
            arm=f'{variant}_{"learned" if learned else "fixed"}'
            renderer=Renderer(base,near).cuda()
            training[arm]=train(renderer,encoder,d['X'][fit],yfit,people[fit],learned,initial[:len(fit)])
            final_checks[arm]=direct_guard(renderer,probe) if encoder is None else run(renderer,encoder,probe,'cuda')
            array=extract(renderer,encoder,d['X'][rows])
            if not np.isfinite(array).all():raise ValueError('invalidfinal features')
            banks[arm]=array; banks[arm+'_raw287']=np.concatenate([array,raw],axis=1)
            if not np.array_equal(banks[arm+'_raw287'][:,-287:],raw):raise ValueError('rawskip changed')
            path=root/f'{arm}.pt'
            with path.open('xb') as out:torch.save(renderer.state_dict(),out)
            checkpoints[arm]=sha(path)
            # Record B1features before deleting renderer; compare through fitted readout below.
            with torch.no_grad():
                single=np.concatenate([encode(renderer,encoder,torch.from_numpy(d['X'][r:r+1]).cuda()).cpu().numpy() for r in fit[:4]])
            training[arm]['B1_source_probe']=single.tolist()
        del encoder
    # Every final featurebank and source-only classifier fit precedes validation labels.
    predictions={}; heads={}
    for arm,array in banks.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=2000))
        model.fit(array[:len(fit)],yfit)
        if int(model[-1].n_iter_.max())>=2000:raise ValueError('source final head convergence failed')
        if arm!='raw287':
            key=arm.removesuffix('_raw287');single=np.array(training[key]['B1_source_probe'])
            if arm.endswith('_raw287'):single=np.concatenate([single,raw[:4]],axis=1)
            if not np.array_equal(model.predict(single),model.predict(array[:4])):raise ValueError('B1/B4 readout argmax changed')
        predictions[arm]=model.predict_proba(array[len(fit):])
        heads[arm]=dict(feature_sha256=digest(array),iterations=int(model[-1].n_iter_.max()),
            coefficient_sha256=digest(model[-1].coef_),scaler_sha256=digest(model[0].scale_))
    for value in training.values():value.pop('B1_source_probe')
    yval=labels(val,len(d['X'])); results={}
    for arm,p in predictions.items():
        result={}
        for person in ('P02','P03'):
            selected=people[val]==person
            result[person]=dict(balanced_accuracy=float(balanced_accuracy_score(yval[selected],p[selected].argmax(1))),
                logloss=float(log_loss(yval[selected],p[selected],labels=[0,1,2])))
        results[arm]=dict(people=result,person_mean_ba=float(np.mean([v['balanced_accuracy'] for v in result.values()])))
    report=dict(source_only=True,P01_labels_opened=False,validation_previously_exposed=True,
        source_sha256=sha(root/'source.sha256'),source=meta,validation_rows_sha256=digest(val),
        source_rows_sha256=digest(fit),source_feature_sha256={k:digest(v) for k,v in banks.items()},
        renderer_checkpoint_sha256=checkpoints,engineering=engineering,projected_seconds=projected,
        training=training,final_checks=final_checks,heads=heads,results=results,reference324=reference(),
        total_seconds=time.monotonic()-started,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
        peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
        no_outer_continuation=True)
    with (root/'results.json').open('x') as out:json.dump(report,out,indent=2)
    with (root/'features.npz').open('xb') as out:np.savez(out,index=rows,**banks)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
