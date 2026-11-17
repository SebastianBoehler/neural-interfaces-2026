"""Train the B1 affine on six source people and verify the physical layer boundary."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from onebn import LearnedBN,Tail,joint,preactivation,source


def plan(rows,d,labels):
    rng=np.random.default_rng(381)
    stream=d['stream'][rows]
    sessions=sorted(set(stream))
    weights=np.empty(len(rows))
    for s in sessions:
        m=stream==s
        for iface in set(d['interface'][rows][m]):
            for y in range(3):
                subset=m&(d['interface'][rows]==iface)&(labels==y)
                assert subset.any()
                weights[subset]=m.sum()/(6*subset.sum())
    episodes=[rng.choice(np.flatnonzero(stream==rng.choice(sessions)),48,replace=False) for _ in range(360)]
    return episodes,torch.tensor(weights,dtype=torch.float32,device='cuda')


def teacher(x,layer):
    # A6 weights the current item and all previous items equally with source count 20.
    mean=x.mean((2,3))
    power=x.square().mean((2,3))
    n=torch.arange(1,len(x)+1,device=x.device,dtype=x.dtype)[:,None]
    m=(20*layer.running_mean+mean.cumsum(0))/(20+n)
    prior=layer.running_var+layer.running_mean.square()
    var=((20*prior+power.cumsum(0))/(20+n)-m.square()).clamp_min(0)
    std=(var+layer.eps).sqrt()
    return .5*((layer.running_var+layer.eps).log()-(var+layer.eps).log()), (layer.running_mean-m)/std


def main(arm,smoke):
    started=time.monotonic()
    assert arm in ('C','D') and 'A100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(381)
    d,people,rows,runtime=source()
    model=runtime.net.eval()
    data=torch.from_numpy(runtime._aligned(rows,True))
    y=joint.labels(rows,len(people))
    episodes,weights=plan(rows,d,y)
    y=torch.from_numpy(y.astype(np.int64)).cuda()
    with torch.no_grad():
        probe=data[:12]
        h=preactivation(model,probe)
        zz=Tail(model)(model.spatial_conv[1](h))
        assert torch.equal(zz,model(probe)), 'A6 tail no longer reproduces raw forward'
        # Only source activations are transferred from the certified CPU front end.
        if smoke:
            use=np.array(sorted(set(np.concatenate(episodes[:2]))))
        else:
            use=np.arange(len(data))
        cache=torch.cat([preactivation(model,data[b]).cuda() for b in np.array_split(use,(len(use)+31)//32)])
    tail=Tail(model).cuda().eval()
    old=model.spatial_conv[1].cuda().eval()
    student=LearnedBN(old).cuda()
    optimizer=torch.optim.AdamW(student.parameters(),lr=.003,weight_decay=.001)
    losses=[]
    checks={}
    for step,b in enumerate(episodes[:2] if smoke else episodes):
        x=cache[np.searchsorted(use,b)]
        z=student.standardized(x)
        desired=teacher(x,old)
        fitted=student.affine(student.encode(z),'real')
        if step==0:
            with torch.no_grad():
                teacher_layer=student.transform(z,*desired)
                err=(tail(teacher_layer)-tail(old(x))).abs().max().item()
                assert err<1e-3 and torch.equal(tail(teacher_layer).argmax(1),tail(old(x)).argmax(1))
                assert fitted[0].abs().max().item()==fitted[1].abs().max().item()==0
                checks['teacher_tail_max_logit_error']=err
                checks['initial_state']=joint.state_sha(student)
        dist=F.mse_loss(fitted[0],desired[0])+F.mse_loss(fitted[1],desired[1])
        actual=tail(student.transform(z,*fitted))
        ce=(F.cross_entropy(actual,y[b],reduction='none')*weights[b]).mean()
        loss=dist if arm=='D' and step<320 else ce
        optimizer.zero_grad()
        loss.backward()
        assert torch.isfinite(loss) and student.read[-1].weight.grad.abs().max()>0
        optimizer.step()
        losses.append([dist.item(),ce.item()])
        if not smoke and ((arm=='D' and step==319) or step==359):
            name='D' if step==319 else ('DC' if arm=='D' else 'C')
            torch.save({k:v.detach().cpu() for k,v in student.state_dict().items()},name+'.pt')
    assert student.threshold.grad.abs().max()>0
    path=Path('smoke.json' if smoke else 'train_'+arm+'.json')
    checkpoints={} if smoke else {a:joint.sha(Path(a+'.pt')) for a in (('C',) if arm=='C' else ('D','DC'))}
    path.write_text(json.dumps(dict(arm=arm,source_sha=joint.sha(Path('source.sha256')),
        train_people=joint.FIT_PEOPLE,train_rows=joint.digest(rows),episodes=joint.digest(np.stack(episodes)),
        held_labels_opened=False,seconds=time.monotonic()-started,smoke=smoke,checks=checks,
        last8_average=np.mean(losses[-8:],axis=0).tolist(),first8_average=np.mean(losses[:8],axis=0).tolist(),
        checkpoints=checkpoints),indent=2))


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2:]==['smoke'])
