"""Original raw source preparation/optimizer recipe, controlled token factorizer."""
import types
import inspect
import numpy as np
import torch
from architecture import TokenLayer
from data354 import original,balanced

WEIGHTS=dict(order=.1,style=.01,contrast=.2,reconstruction=.02)


def attach(model,arm):
    trace=dict(arm=arm,steps=0,plans=[],auxiliary={},gradient_max=0.,mixture_gradient_max=0.,gate_variation_max=0.)
    def train(self,X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only=False,avg=None):
        if weighted or head_only or avg is not None or self.aug or self.rot or self.align:
            raise ValueError('registered strong raw recipe changed')
        if original.sha(inspect.getfile(type(self.net)))!='604e690ad1743a8dc9eeb87c0c3065d28619cbd4dbda8c9790e60924cafbe81a':
            raise ValueError('actual FBC backbone source changed')
        self._prepare();trace['initial_backbone_sha256']=original.state_sha(self.net)
        if arm in ('M','L','G'):
            before=torch.get_rng_state().clone()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(354001);self.net.temporal_layer=TokenLayer(self.net.temporal_layer,arm)
            if not torch.equal(before,torch.get_rng_state()):raise ValueError('adapter changed backbone RNG')
        adapter=self.net.temporal_layer.adapter if arm in ('M','L','G') else None
        trace['adapter_parameters']=0 if adapter is None else sum(p.numel() for p in adapter.parameters())
        trace['trainable_parameters']=sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        from deep import param_groups
        groups=param_groups(self.net,lr,self.lld,False)
        optimizer=torch.optim.AdamW(groups,lr=lr,weight_decay=self.wd)
        count=len(self._batches(rows,np.random.default_rng(0)));visited=set()
        head=self.net.final_layer.parametrizations.weight.original
        if sum(p is head for group in groups for p in group['params'])!=1:
            raise ValueError('actual constrained head not owned exactly once')
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,[g['lr'] for g in groups],total_steps=epochs*count)
        for epoch in range(epochs):
            batches=self._batches(rows,rng) if arm=='O' else balanced(self,rows,rng,count)
            trace['plans'].append(original.digest(np.concatenate([self._order[b] for b in batches])))
            if len(batches)!=count:raise ValueError('step control mismatch')
            for indices in batches:
                visited.update(map(int,indices))
                optimizer.zero_grad();logits=self.net(X[indices])[:,:3];loss=loss_fn(logits,y[indices])
                if adapter is not None:
                    terms,variation=adapter.losses(self._grp[torch.from_numpy(indices)])
                    for name,value in terms.items():
                        if not torch.isfinite(value):raise ValueError('nonfinite auxiliary loss')
                        loss=loss+WEIGHTS[name]*value
                        trace['auxiliary'][name]=trace['auxiliary'].get(name,0.)+float(value.detach())
                    trace['gate_variation_max']=max(trace['gate_variation_max'],float(variation.detach()))
                loss.backward()
                gradients=[p.grad for p in self.net.parameters() if p.grad is not None]
                if not all(torch.isfinite(g).all() for g in gradients):raise ValueError('nonfinite gradient')
                trace['gradient_max']=max(trace['gradient_max'],max(float(g.abs().max()) for g in gradients))
                if arm=='G':
                    values=[float(p.grad.abs().max()) for n,p in adapter.named_parameters() if 'generator' in n and p.grad is not None]
                    trace['mixture_gradient_max']=max(trace['mixture_gradient_max'],max(values))
                optimizer.step();scheduler.step();trace['steps']+=1
                if adapter is not None:adapter.cache=None
            if epoch%10==0:print('epoch',epoch,'arm',arm,'steps',trace['steps'],flush=True)
        if trace['steps']!=epochs*count or trace['gradient_max']<=0:raise ValueError('inactive or incomplete fit')
        if arm in ('L','G') and (set(trace['auxiliary'])!=set(WEIGHTS) or min(trace['auxiliary'].values())<=0):
            raise ValueError('auxiliary silent no-op')
        if arm=='G' and (trace['mixture_gradient_max']<=1e-12 or trace['gate_variation_max']<=1e-8):
            raise ValueError('dynamic transition mixture silent no-op')
        if epochs==40 and len(visited)!=len(rows):raise ValueError('source rows never trained')
        trace['visited_source_rows']=len(visited)
        trace.update(epochs=epochs,batches_per_epoch=count,scale=self.scale,source_order_sha256=original.digest(self._order),
                     final_state_sha256=original.state_sha(self.net))
    model._train=types.MethodType(train,model)
    return trace
