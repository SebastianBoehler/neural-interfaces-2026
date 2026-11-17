"""Copy-onlyJS outer intervention; original inner member/gate/label path remains intact."""
import numpy as np

from .v6chain import Ensemble6
from .nets import NetBank, SharedFIRSJit


def js_outer_kernel(base,removed,added,fbl,unit):
    """base includes the old removed component; add one complete adaptedFBLunit."""
    denominator=1.+float(unit)
    result=(base+unit*(added-removed+fbl))/denominator
    if not np.isfinite(result).all() or (result<0).any() or not np.allclose(result.sum(-1),1.,atol=1e-10):
        raise ValueError('JS outer mixture is not normalized finite probabilities')
    return result


def build_js_bank(z,dev,root):
    files=[str(f) for f in z['js_cc_files']]
    if len(files)!=6 or len(set(files))!=6 or list(z['js_seeds'])!=list(range(6)):
        raise ValueError('actual six distinct seed identities required')
    p={k:z['js_'+k] for k in ('align','k','G','scale','shrink')}
    bank=NetBank([dict(p,sjit_file=np.array(f)) for f in files],np.ones(6,dtype=bool),dev,root)
    if len(bank.groups)!=1:raise ValueError('one identical causal input group required')
    stream,members=bank.groups[0]
    bank.groups=[(stream,[(i,SharedFIRSJit(dict(p,sjit_file=np.array(files[i])),dev,root)) for i,_ in members])]
    if [i for i,_ in bank.groups[0][1]]!=list(range(6)):raise ValueError('shared FIR source order differs')
    return bank


class EnsembleJS(Ensemble6):
    def __init__(self,z,dev,root):
        if 'js_on' not in z or int(z['js_on']) not in (0,1):
            raise ValueError('explicit registered js_on0/1 required')
        self.js_on=bool(int(z['js_on']));self.js_bank=None
        super().__init__(z,dev,root)
        if self.js_on:
            if (self.routed,self.v6r,self.r15)!=(1,False,True):
                raise ValueError('JS supports only the pinned cap250 original label/gate path')
            files=[str(f) for f in z['js_cc_files']]
            if len(files)!=6 or len(set(files))!=6 or list(z['js_seeds'])!=list(range(6)):
                raise ValueError('actual six distinct seed identities required')
            p={k:z['js_'+k] for k in ('align','k','G','scale','shrink')}
            self.js_bank=build_js_bank(z,dev,root)
            fb=self.w[1+self.ft.slots['fb']].sum();fbl=self.w[1+self.ft.slots['fbl']].sum()
            if not np.isclose(fb,fbl,rtol=0,atol=1e-15):raise ValueError('registered unit FB/FBL weights differ')
            self.js_unit=(1-self.w_rest)*fb
            self.routeB.a=self.w[0]*(1-self.w_rest)/(1+self.js_unit)
            if not np.isclose(self.js_unit,1/16,atol=1e-15) or not np.isclose(self.routeB.a,1/17,atol=1e-15):
                raise ValueError('registered17unit/core1 mapping differs')
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        if self.js_bank is not None:self.js_bank.reset_state()
        self.js_p=None;self.js_calls=0

    def step(self,x):
        if self.js_on:
            values=self.js_bank(x)
            if [i for i,_ in values]!=list(range(6)):raise ValueError('CC30 bank source order differs')
            p=np.mean([p for _,p in values],axis=0)
            p=np.clip(p,1e-6,None);self.js_p=p/p.sum();self.js_calls+=1
        out=super().step(x)
        if self.js_on:self.trace['js']=dict(cc=self.js_p.copy(),calls=self.js_calls,unit=self.js_unit)
        return out

    def mix6(self,mix,members,p_ft):
        if not self.js_on:return super().mix6(mix,members,p_ft)
        if self.js_p is None:raise ValueError('JS outer bank was not run for current window')
        # Keep original FT computation/labels, but do not inject oldFBk2delta into the outer mix.
        base=super().mix6(mix,members,{s:p for s,p in p_ft.items() if s!='fb'})
        fb_ids=self.ft.slots['fb'];fw=self.w[1+fb_ids]
        removed=(fw[:,None]*np.stack([members[i] for i in fb_ids])).sum(0)/fw.sum()
        ids=self.ft.slots['fbl'];weights=self.w[1+ids]
        fbl=(weights[:,None]*np.stack([members[i] for i in ids])).sum(0)/weights.sum()
        if 'fbl' in p_ft:
            fbl=np.clip(fbl+p_ft['fbl']-members[self.ft.member['fbl']],1e-6,None);fbl/=fbl.sum()
        return js_outer_kernel(base,removed,self.js_p,fbl,self.js_unit)
