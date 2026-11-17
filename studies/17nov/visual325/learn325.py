"""Person-crossfitted heads and bounded display learning; fixed source epochs only."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from support325 import features, digest, state_sha


def batches(people, seed):
    rng = np.random.default_rng(seed)
    chunks = []
    for person in np.unique(people):
        rows = rng.permutation(np.flatnonzero(people == person))
        chunks += [(person, r) for r in np.array_split(rows, max(1,(len(rows)+3)//4))]
    return [chunks[i] for i in rng.permutation(len(chunks))]


def encode(renderer, encoder, x):
    return renderer.direct(x) if encoder is None else features(encoder,renderer(x),1435)


def extract(renderer, encoder, array, device='cuda'):
    values = []
    with torch.no_grad():
        for start in range(0,len(array),4):
            values.append(encode(renderer,encoder,torch.from_numpy(array[start:start+4]).to(device)).cpu().numpy())
    return np.concatenate(values)


def train(renderer, encoder, x, y, people, learned, initial):
    torch.manual_seed(325)
    names = sorted(set(people)); keys = {p:i for i,p in enumerate(names)}
    heads = nn.ModuleList([nn.Linear(initial.shape[1],3) for _ in names]).cuda()
    optimizer = torch.optim.AdamW(heads.parameters(),lr=.001,weight_decay=.01)
    display_optimizer = torch.optim.Adam(renderer.parameters(),lr=.003)
    class_weights = [torch.as_tensor(1/np.bincount(y[people!=p],minlength=3),dtype=torch.float32,device='cuda') for p in names]
    class_weights = [v/v.mean() for v in class_weights]
    state = None if encoder is None else state_sha(encoder)
    snapshots = {n:p.detach().cpu().clone() for n,p in renderer.named_parameters()}
    trace=[]; order_hashes=[]; first_gradients=None; first_parameter_change=None
    started=time.monotonic()
    for epoch in range(-1,3):
        losses=[]; schedule=batches(people,325+epoch)
        order_hashes.append(digest(np.concatenate([r for _,r in schedule])))
        for person,rows in schedule:
            target=torch.as_tensor(y[rows],dtype=torch.long,device='cuda'); own=keys[person]
            if epoch<0 or not learned:
                z=torch.as_tensor(initial[rows],device='cuda')
            else:
                z=encode(renderer,encoder,torch.from_numpy(x[rows]).cuda())
            normalized=F.layer_norm(z,(z.shape[1],))
            optimizer.zero_grad(set_to_none=True)
            head_loss=sum(F.cross_entropy(heads[i](normalized.detach()),target,weight=class_weights[i]) for i in range(7) if i!=own)/6
            head_loss.backward(); optimizer.step()
            if learned and epoch>=0:
                display_optimizer.zero_grad(set_to_none=True)
                # Detached held-person head: its parameters have never trained on this person.
                logits=F.linear(normalized,heads[own].weight.detach(),heads[own].bias.detach())
                loss=F.cross_entropy(logits,target,weight=class_weights[own])
                if not torch.isfinite(loss): raise ValueError('display loss nonfinite')
                loss.backward()
                gradients={n:None if p.grad is None else float(p.grad.abs().max()) for n,p in renderer.named_parameters()}
                if any(v is None or not np.isfinite(v) or v<=1e-12 for v in gradients.values()):
                    raise ValueError('true-label display trainstep has inactive parameters')
                if first_gradients is None: first_gradients=gradients
                torch.nn.utils.clip_grad_norm_(renderer.parameters(),1.)
                display_optimizer.step(); losses.append(float(loss.detach()))
                if first_parameter_change is None:
                    first_parameter_change={n:float((p.detach().cpu()-snapshots[n]).abs().max()) for n,p in renderer.named_parameters()}
                    if min(first_parameter_change.values())<=0: raise ValueError('renderer optimizer no-op')
        trace.append(dict(epoch=epoch,display_loss=None if not losses else float(np.mean(losses))))
    if encoder is not None and (state_sha(encoder)!=state or any(p.grad is not None for p in encoder.parameters())):
        raise ValueError('frozen encoder changed during true training')
    delta={n:float((p.detach().cpu()-snapshots[n]).abs().max()) for n,p in renderer.named_parameters()}
    if learned and min(delta.values())<=0: raise ValueError('learned display inactive')
    if not learned and max(delta.values())!=0: raise ValueError('fixed renderer changed')
    return dict(seconds=time.monotonic()-started,epochs=3,warmup_epochs=1,seed=325,
        display_trace=trace,first_gradients=first_gradients,first_parameter_change=first_parameter_change,
        parameter_delta=delta,batch_order_sha256=order_hashes,head_excluded_people=names,
        encoder_sha256=state,heads_sha256=state_sha(heads))
