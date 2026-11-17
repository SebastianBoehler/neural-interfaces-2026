"""Same-weight SharedFIR tails read the original bank's current input, without mutating that bank."""
import numpy as np
import torch

from .adapter_shared import EnsembleJS
from .inputs import input_key
from .v6chain import Ensemble6


class ReuseEnsemble(EnsembleJS):
    def __init__(self,z,dev,root):
        super().__init__(z,dev,root);self.js_group_id=None
        if self.js_on:
            p={k:z['js_'+k] for k in ('align','k','G','scale','shrink')};matches=[]
            for index,(stream,members) in enumerate(self.nets.groups):
                if input_key({k:getattr(stream,k) for k in p})==input_key(p):matches.append((index,members))
            if len(matches)!=1 or 35 not in [i for i,_ in matches[0][1]] or self.ft.member['fb']!=35:
                raise ValueError('exact original net35 input group/current FT member required')
            self.js_group_id=matches[0][0]

    def reset_state(self):
        super().reset_state();self.js_step=0;self.js_done=0;self.js_reuse=True

    @torch.no_grad()
    def consume_current_input(self):
        if self.js_done==self.js_step or self.js_step<1:raise ValueError('duplicate or premature JS callback')
        original=self.nets.groups[self.js_group_id][0]
        if original.n!=self.js_step or self.js_bank.groups[0][0].n!=0:
            raise ValueError('stale original input or auxiliary EA stream unexpectedly advanced')
        h=self.nets.last_inputs[35]
        if h.shape!=(41,480) or h.dtype!=np.float32 or not np.isfinite(h).all():raise ValueError('current original input invalid')
        H=torch.from_numpy(h[None]);cache={};inputs={};probs=[]
        for i,net in self.js_bank.groups[0][1]:
            if i!=len(probs):raise ValueError('six-seed tail order changed')
            if net.dtype not in inputs:inputs[net.dtype]=H.to(dtype=net.dtype).to(self.js_bank.dev)
            probs.append(net.proba(inputs[net.dtype],cache))
        p=torch.stack(probs).cpu().numpy()[:,0].mean(0);p=np.clip(p,1e-6,None)
        self.js_p=p/p.sum();self.js_calls+=1;self.js_done=self.js_step

    def mix6(self,mix,members,p_ft):
        if self.js_on:self.consume_current_input()
        return super().mix6(mix,members,p_ft)

    def step(self,x):
        if self.js_on:self.js_step+=1;self.js_p=None
        out=Ensemble6.step(self,x)  # Original bank/inner/FT path; mix6 consumes only after the original current input exists.
        if self.js_on:
            if self.js_done!=self.js_step or self.js_calls!=self.js_step:raise ValueError('JS callback missed the current window')
            self.trace['js']=dict(cc=self.js_p.copy(),calls=self.js_calls,unit=self.js_unit,input_reuse=True,source_step=self.js_step)
        return out
