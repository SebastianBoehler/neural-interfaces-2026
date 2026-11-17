"""Four fixed matched-exposure pretexts; original native40 supervision follows unchanged."""
import copy
import time
import types
import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized
from common344 import state_sha,digest,GROUPS


def attach(model,arm,teacher,fit,continuation,steps=64):
    supervised=model._train;record={'arm':arm,'steps':0,'losses':[],'body_grad_max':[]};fine_losses=[]
    def train(self,X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only=False,avg=None):
        if weighted or epochs!=40 or loss_fn.weight is not None:raise ValueError('native final recipe changed')
        record['initial_native_state_sha256']=state_sha(self.net);self._prepare();head=self.net.final_layer
        if not is_parametrized(head,'weight') or head.weight.shape!=(3,1152):raise ValueError('native classifier changed')
        original_head=state_sha(head);torch_state=torch.get_rng_state().clone();numpy_state=copy.deepcopy(rng.bit_generator.state);gen_state=gen.get_state().clone()
        width=287 if arm=='moment' else 44239
        auxiliary=None
        if arm!='extra_supervised':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(337);auxiliary=nn.Linear(1152,width)
            if not torch.equal(torch_state,torch.get_rng_state()):raise ValueError('auxiliary RNG contamination')
        record['auxiliary_parameters']=0 if auxiliary is None else sum(p.numel() for p in auxiliary.parameters())
        record['loss_dimensions']=3 if auxiliary is None else width
        pos=np.full(len(self.d['X']),-1,int);pos[self._order]=np.arange(len(self._order));tpos=np.full(len(self.d['X']),-1,int);tpos[teacher['rows']]=np.arange(len(teacher['rows']))
        # Preserve the EXACT exposed target multiset, including padded/repeated rows.
        exposed=teacher['trace'].ravel();random=np.random.default_rng(337);shuffled=None
        for attempt in range(128):
            candidate=exposed[random.permutation(len(exposed))]
            if np.all(candidate!=exposed):shuffled=candidate.reshape(teacher['trace'].shape);break
        if shuffled is None or not np.array_equal(np.sort(shuffled.ravel()),np.sort(exposed)):raise ValueError('exposure-matched target derangement')
        record['shuffle_trace_sha256']=digest(shuffled)
        groups=(np.arange(41),np.arange(41,287)) if arm=='moment' else GROUPS
        targets=teacher['moment' if arm=='moment' else 'covariance'];tap={}
        def hook(module,args):tap['features']=args[0]
        handle=head.register_forward_pre_hook(hook);parameters=list(self.net.parameters())+([] if auxiliary is None else list(auxiliary.parameters()))
        opt=torch.optim.AdamW(parameters,lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=steps)
        body=[p for n,p in self.net.named_parameters() if p.requires_grad and 'final_layer' not in n];before=[p.detach().clone() for p in body]
        record['torch_rng_pretext_before']=digest(torch_state.numpy());record['pretext_rows_sha256']=digest(teacher['trace'][:steps]);tick=time.monotonic();self.net.train()
        try:
            for batch_index,batch in enumerate(teacher['trace'][:steps]):
                j=pos[batch]
                if (j<0).any():raise ValueError('pretext input outside source6')
                opt.zero_grad();logits=self.net(X[j])
                if arm=='extra_supervised':loss=loss_fn(logits,y[j]);parts=[float(loss.detach())]
                else:
                    targetrows=shuffled[batch_index] if arm=='shuffled' else batch
                    if (targetrows<0).any() or not np.isin(targetrows,fit).all():raise ValueError('pretext target outside source6')
                    actual=torch.from_numpy(targets[tpos[targetrows]]);error=(auxiliary(tap['features'])-actual).square()
                    losses=torch.stack([error[:,g].mean() for g in groups]);loss=losses.mean();parts=losses.detach().tolist()
                loss.backward();grads=[p.grad for p in body if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or max(float(g.abs().max()) for g in grads)<=0:raise ValueError('pretext body inactive/nonfinite')
                if auxiliary is not None and any(p.grad is not None for p in head.parameters()):raise ValueError('unsupervised pretext touches native head')
                record['body_grad_max'].append(max(float(g.abs().max()) for g in grads));record['losses'].append(parts);opt.step();sched.step();record['steps']+=1
        finally:handle.remove()
        record['body_change_max']=max(float((p.detach()-old).abs().max()) for p,old in zip(body,before));record['pretext_seconds']=time.monotonic()-tick
        record['torch_rng_pretext_after']=digest(torch.get_rng_state().numpy());record['native_head_unchanged_by_pretext']=state_sha(head)==original_head
        if record['steps']!=steps or record['body_change_max']<=0 or (auxiliary is not None and not record['native_head_unchanged_by_pretext']):raise ValueError('pretext completeness/headguard')
        torch.set_rng_state(torch_state);rng.bit_generator.state=numpy_state;gen.set_state(gen_state)
        record['fine_rng_restored']=digest(torch.get_rng_state().numpy())==record['torch_rng_pretext_before']
        def loss_hook(module,args,value):fine_losses.append(float(value.detach()))
        loss_handle=loss_fn.register_forward_hook(loss_hook)
        try:continuation(self,auxiliary,supervised,X,y,sw,rows,epochs,lr,rng,gen,loss_fn,weighted,head_only,avg)
        finally:loss_handle.remove()
        record['fine_losses']=fine_losses
    model._train=types.MethodType(train,model)
    return record
