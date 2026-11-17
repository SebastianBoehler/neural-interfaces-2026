"""Fixed40epoch source-only training; source losses/recalls, no query selection."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import Decoder
from activity331 import fingerprint
from data331 import digest

MODES={'L':'local','F':'static','S':'selective'}


def make(arm,mean,sd):
    torch.manual_seed(331)
    return Decoder(MODES[arm],mean,sd).float()


def sample_weights(y,people):
    result=np.zeros(len(y),np.float32)
    for person in np.unique(people):
        for k in (0,1,2):
            mask=(people==person)&(y==k)
            if not mask.any():raise ValueError('sourceperson missingclass')
            result[mask]=1/mask.sum()
    return result/result.mean()


def schedule(n,epoch):return np.random.default_rng(331+epoch).permutation(n)


def fit(arm,x,y,people,mean,sd):
    model=make(arm,mean,sd);initial={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40,eta_min=.00001)
    weights=sample_weights(y,people);traces=[];orders=[];started=time.monotonic();activity={}
    for epoch in range(40):
        model.train();rows=schedule(len(x),epoch);orders.append(digest(rows));loss_sum=weight_sum=0.;confusion=np.zeros((3,3))
        gradient_max={n:0. for n,_ in model.named_parameters()}
        for start in range(0,len(rows),32):
            r=rows[start:start+32];xx=torch.from_numpy(x[r]);yy=torch.from_numpy(y[r]);ww=torch.from_numpy(weights[r])
            optimizer.zero_grad(set_to_none=True);logits=model(xx)
            ce=F.cross_entropy(logits,yy,reduction='none');loss=(ce*ww).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite source loss')
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise ValueError('nonfinite train gradient:'+name)
                    gradient_max[name]=max(gradient_max[name],float(p.grad.abs().max()))
            optimizer.step();loss_sum+=float((ce.detach()*ww).sum());weight_sum+=float(ww.sum())
            pred=logits.detach().argmax(1).numpy()
            for k in (0,1,2):
                for j in (0,1,2):confusion[k,j]+=float(weights[r][(y[r]==k)&(pred==j)].sum())
        scheduler.step();recall=np.diag(confusion)/confusion.sum(1)
        traces.append(dict(epoch=epoch+1,weighted_source_CE=loss_sum/weight_sum,online_source_class_recalls=recall.tolist(),lr=optimizer.param_groups[0]['lr']))
        activity[str(epoch+1)]=gradient_max
        print(f'{arm}sourceepoch{epoch+1}/40 loss={loss_sum/weight_sum:.6f}',flush=True)
    changes={n:float((p.detach()-initial[n]).abs().max()) for n,p in model.named_parameters()}
    for name,delta in changes.items():
        inactive=('.dt.' in name and arm!='S') or (name.endswith('.frequency') and arm=='L')
        if inactive and delta!=0:raise ValueError('intentionallyinactive parameterchanged:'+name)
        if not inactive and (not np.isfinite(delta) or delta<=0):raise ValueError('trainedparameter no-op:'+name)
    return model.eval(),dict(arm=arm,seed=331,epochs=40,source_trace=traces,gradient_activity=activity,
        parameter_changes=changes,batch_order_sha256=orders,seconds=time.monotonic()-started,model_sha256=fingerprint(model))


def predict(model,x):
    with torch.no_grad():return np.concatenate([model(torch.from_numpy(x[i:i+1])).softmax(1).numpy() for i in range(len(x))])
