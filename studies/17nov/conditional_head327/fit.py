"""Loss-calibrated source-only conditional-logit follow-up to the locked §326 pilot."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch

OLD = Path('/mnt/lustre/work/gehler/gfh527/ni26_s326_conditional_head_20261117')
sys.path.insert(0,str(OLD/'studies/17nov/conditional_head'))
from prepare import QUERY, ORDERS, sha, Native
from pilot import native_fit
from loss import objective, penalty


def source_margin(x, y, theta):
    logits = x @ theta[:,:-1].T + theta[:,-1]
    true = logits.gather(1, y[:,None]).squeeze(1)
    wrong = logits.clone(); wrong.scatter_(1, y[:,None], -torch.inf)
    return float((true-torch.logsumexp(wrong,dim=1)).mean().item())


def main():
    started = time.monotonic(); previous = OLD/'runs/s326'; out = Path('runs/s327')
    out.mkdir(parents=True, exist_ok=False)
    if sha(OLD/'source.sha256') != 'f35b41103b37098de243e0a59312b1f54dc76bc4a2959d22f7dba1aeb55b811d':
        raise ValueError('changed §326 source lock')
    prep = json.loads((previous/'prepare.json').read_text()); old = json.loads((previous/'pilot-guards.json').read_text())
    if sha(previous/'prepare.json') != old['prepare_sha256'] or sha(previous/'train.npz') != prep['train_sha256']:
        raise ValueError('changed preparation')
    if sha(previous/'pilot-predictions.npz') != old['predictions_sha256'] or old['query_labels_read_for_accuracy'] is not False:
        raise ValueError('changed/no-score guard or archived zero predictions')
    for name,digest in prep['streams_sha256'].items():
        if sha(previous/name)!=digest: raise ValueError(f'changed stream {name}')
    if len(prep['streams_sha256'])!=16 or prep['fit_people'] != [f'P{i:02d}' for i in range(4,11)]:
        raise ValueError('source matrix incomplete or wrong fit people')
    with np.load(previous/'train.npz',allow_pickle=False) as z:
        x = np.ascontiguousarray(z['x']); y = np.ascontiguousarray(z['y']); D = np.ascontiguousarray(z['D'])
        mean = z['mean'].copy(); scale = z['scale'].copy()
    if not torch.cuda.is_available(): raise RuntimeError('A100 required; no fallback')
    device = torch.device('cuda:0'); xd = torch.as_tensor(x,device=device); yd = torch.as_tensor(y,device=device)
    dd = torch.as_tensor(D,device=device); native = Native()
    zero,L0,grad,calls = native_fit(x,y); cpu_loss,cpu_grad = native.objective(x,y,zero)
    if abs(L0-old['diagnostics']['zero']['loss'])>1e-9 or abs(L0-cpu_loss)>1e-9 or grad>1e-5:
        raise ValueError('native zero objective differs from §326')
    w0=torch.as_tensor(zero,device=device).clone().requires_grad_(True); q=float(penalty(w0,dd).item())
    if abs(q-old['q'])>1e-9 or q<=1e-20 or L0<=0: raise ValueError('q/baseline objective changed')
    zero_loss=objective(xd,yd,w0,dd,0,q); auto=torch.autograd.grad(zero_loss,w0)[0]
    if abs(zero_loss.item()-L0)>1e-9 or np.max(np.abs(auto.cpu().numpy()-cpu_grad))>1e-8:
        raise ValueError('zero loss/gradient mismatch')
    pg=torch.autograd.grad(penalty(w0,dd),w0)[0]; direction=pg/torch.linalg.vector_norm(pg); eps=1e-5
    fd=((penalty(w0+eps*direction,dd)-penalty(w0-eps*direction,dd))/(2*eps)).item()
    if not np.isclose(fd,torch.linalg.vector_norm(pg).item(),atol=1e-8,rtol=1e-5):
        raise ValueError('real-data finite difference failed')
    zd=penalty(w0,torch.zeros_like(dd)); zg=torch.autograd.grad(zd,w0)[0]
    if zd.item()!=0 or torch.count_nonzero(zg): raise ValueError('zero D not identity')
    common=torch.arange(x.shape[1],device=device,dtype=torch.float64)*1e-10
    shifted=torch.cat((w0[:,:-1]+common,w0[:,-1:]),dim=1)
    if not torch.allclose(penalty(shifted,dd),penalty(w0,dd),atol=1e-12,rtol=1e-10):
        raise ValueError('softmax gauge invariance failed')
    models={'zero':zero}; diagnostics={'zero':{'loss':L0,'gradmax':grad,'calls':calls,
                       'penalty_ratio':1.,'source_margin':source_margin(xd,yd,w0),'weight_delta':0.}}
    for fraction,name in ((.01,'mild'),(.1,'moderate')):
        lam=fraction*L0
        if not np.isclose((lam*penalty(w0,dd)/q).item(),fraction*L0,rtol=1e-12,atol=1e-15):
            raise ValueError('initial penalty scale incorrect')
        theta=torch.zeros_like(w0,requires_grad=True)
        opt=torch.optim.LBFGS([theta],lr=1.,max_iter=2000,max_eval=2500,history_size=10,
                              tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        count=[0]
        def closure():
            opt.zero_grad(); value=objective(xd,yd,theta,dd,lam,q); value.backward(); count[0]+=1
            if not torch.isfinite(value) or not torch.isfinite(theta.grad).all(): raise ValueError('nonfinite fit')
            return value
        opt.step(closure); value=closure(); maxgrad=float(theta.grad.abs().max().item())
        if maxgrad>1e-5: raise ValueError(f'{name} nonconverged {maxgrad}')
        models[name]=theta.detach().cpu().numpy()
        diagnostics[name]={'coefficient':lam,'initial_penalty_fraction':fraction,'loss':float(value.item()),
                           'gradmax':maxgrad,'calls':count[0], 'penalty_ratio':float(penalty(theta,dd).item()/q),
                           'source_margin':source_margin(xd,yd,theta),
                           'weight_delta':float(torch.linalg.vector_norm(theta-w0).item())}
    output={}; changed={'mild':0,'moderate':0}
    with np.load(previous/'pilot-predictions.npz',allow_pickle=False) as old_predictions:
        for person in QUERY:
            for order in ORDERS:
                with np.load(previous/f'{person}_{order}.npz',allow_pickle=False) as z:
                    rows=z['rows'].copy(); features=(z['features']-mean)/scale
                for name, theta in models.items():
                    logits=features@theta[:,:-1].T+theta[:,-1]
                    if not np.isfinite(logits).all(): raise ValueError('nonfinite query logits')
                    pred=np.argmax(logits,axis=1); key=f'{person}|{order}|{name}'; output[key]=pred
                    if name=='zero' and not np.array_equal(pred,old_predictions[key]):
                        raise ValueError(f'zero query identity failed {person}/{order}')
                    if name!='zero': changed[name]+=int(np.count_nonzero(pred!=output[f'{person}|{order}|zero']))
                output[f'{person}|rows']=rows
    if any(n==0 for n in changed.values()): raise ValueError(f'inactive query prediction change: {changed}')
    with (out/'predictions.npz').open('xb') as f: np.savez(f,**output)
    metadata={'prepare_sha256':sha(previous/'prepare.json'),'old_guards_sha256':sha(previous/'pilot-guards.json'),
              'source_train_sha256':prep['train_sha256'],'prediction_sha256':sha(out/'predictions.npz'),
              'q':q,'L0':L0,'diagnostics':diagnostics,'changed_labels':changed,'finite_difference':fd,
              'device':torch.cuda.get_device_name(),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
              'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_s':time.monotonic()-started,
              'query_labels_read_for_accuracy':False}
    with (out/'guards.json').open('x') as f: json.dump(metadata,f,indent=2,sort_keys=True)
    print(json.dumps(metadata,sort_keys=True),flush=True)


if __name__=='__main__': main()
