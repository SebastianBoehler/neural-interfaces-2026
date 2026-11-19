"""Instrument and locally substitute only the owning native CE weight tensor."""

from hashlib import sha256
import inspect
from textwrap import dedent
from types import MethodType

import numpy as np
import torch

from settings import stamp
from weights26 import multipliers,preserve,rotate


NAMES=("fb","fbl","fbraw","sinc")
MARK="    for s, m in self.nets.items():\n"


def patched_loop(bound):
    code=dedent(inspect.getsource(bound.__func__))
    assert code.count(MARK)==1 and code.startswith("def _refit(self, t, lab, wt):")
    assert code.count('ce = (wt * F.cross_entropy(za, lab, reduction="none")).sum() / wt.sum()')==1
    altered=code.replace(MARK,"    original_weights = wt\n"+MARK+"        wt = self.f26_weights(s, original_weights)\n")
    namespace=dict(bound.__func__.__globals__)
    exec(compile(altered,"<F-C26 actual FineTune._refit>","exec"),namespace)
    return MethodType(namespace["_refit"],bound.__self__)


class Refit:
    def __init__(self,model,arm):
        self.model,self.arm=model,arm
        self.ft=model.ft
        assert tuple(sorted(self.ft.member))==NAMES and self.ft.on
        assert model.routed==1 and model.r15 and not model.v6r
        assert (self.ft.t0,self.ft.every,self.ft.n_steps,self.ft.cap)==(20,10,5,250)
        self.groups=tuple(tuple(self.ft.slots[name]) for name in NAMES)
        assert all(len(g)==6 for g in self.groups) and len(set(sum(self.groups,())))==24
        self.base=tuple(self.model.w[1+np.asarray(g)] for g in self.groups)
        self.raw_hash={name:sha256() for name in NAMES}
        self.past=[]
        self.events=[]
        self.protected=[]
        self.warm=[]
        self.pending=self.old_weights=None
        self.called=[]
        self.native=self.ft._refit
        self.call=self.native if arm=="stock" else patched_loop(self.native)
        self.ft._refit=self.monitor
        if arm!="stock":
            self.ft.f26_weights=self.use

    def use(self,name,original):
        assert name in NAMES and name not in self.called and self.old_weights is not None
        assert original.dtype==torch.float32 and tuple(original.shape)==self.old_weights.shape
        self.called.append(name)
        if self.arm=="identity":
            return original
        assert self.pending is not None
        exact=torch.as_tensor(self.pending[:,NAMES.index(name)].copy(),device=self.ft.dev)
        assert exact.dtype==original.dtype and exact.shape==original.shape
        return exact

    def generators(self):
        return {name:stamp(self.ft.gens[name].get_state().cpu().numpy()) for name in NAMES}

    def monitor(self,age,labels,weights):
        ft,model=self.ft,self.model
        assert len(self.past)==len(ft.lab)==age and model.log6["n"]==age+1 and not model.recB
        assert ft.relabels==ft.refits+1 and set(len(v) for v in ft.X.values())=={min(age+1,ft.cap)}
        y=np.asarray(labels,dtype=np.int64)
        w=np.asarray(weights,dtype=np.float32)
        assert y.shape==w.shape==(age,) and np.isin(y,(0,1,2)).all()
        assert np.isfinite(w).all() and (w>0).all()
        q=np.asarray(self.past)
        data=dict(age=age,labels=stamp(y),confidence=stamp(w),gen_before=self.generators(),
                  frozen=stamp(q),raw={name:h.hexdigest() for name,h in self.raw_hash.items()})
        self.old_weights=w
        self.called=[]
        measure=dict(mass_error=0.,max_weight_move=0.,changed=0,between_family=0.,factor_min=1.,factor_max=1.)
        if self.arm in ("peer","sham"):
            factor=multipliers(q,y)
            if self.arm=="sham":
                factor=rotate(factor,y)
            self.pending,gap=preserve(factor,w,y)
            diff=np.abs(self.pending-w[:,None])
            measure=dict(mass_error=gap,max_weight_move=float(diff.max()),changed=int(np.count_nonzero(diff)),
                         between_family=float(np.ptp(self.pending,axis=1).max()),factor_min=float(factor.min()),
                         factor_max=float(factor.max()))
        before=(ft.steps,ft.refits)
        self.call(age,labels,weights)
        assert (ft.steps,ft.refits)==(before[0]+20,before[1]+1)
        assert sorted(self.called)==([] if self.arm=="stock" else list(NAMES))
        self.events.append(dict(**data,gen_after=self.generators(),calls=len(self.called),**measure))
        self.pending=self.old_weights=None

    def row(self):
        model,ft=self.model,self.ft
        trace=model.trace
        p=trace["v6"]
        self.protected.append(np.stack((trace["mix"],trace["shadow"],trace["out_v5"],p["label"])))
        self.warm.append(np.stack([p["p_ft"].get(name,np.zeros(3)) for name in NAMES]))
        if model.recB or len(self.past)>=ft.cap:
            return
        age=len(self.past)
        assert model.log6["n"]==age+1
        for name in NAMES:
            actual=ft.X[name][-1]
            np.testing.assert_array_equal(actual,p["ft_inputs"][ft.member[name]])
            self.raw_hash[name].update(np.ascontiguousarray(actual).tobytes())
        current=np.stack([(w[:,None]*np.stack([p["members"][i] for i in g])).sum(0)/w.sum()
                          for g,w in zip(self.groups,self.base,strict=True)])
        np.testing.assert_allclose(current.sum(1),1.,atol=1e-12,rtol=0)
        self.past.append(current)

    def summary(self):
        e=self.events
        return dict(refits=len(e),active=sum(r["changed"]>0 for r in e),touched=sum(r["changed"] for r in e),
                    mass_error=max((r["mass_error"] for r in e),default=0.),
                    family_difference=max((r["between_family"] for r in e),default=0.),
                    max_weight_move=max((r["max_weight_move"] for r in e),default=0.),
                    factors=[min((r["factor_min"] for r in e),default=1.),max((r["factor_max"] for r in e),default=1.)],
                    consumes=sum(r["calls"] for r in e),rows=sum(r["age"] for r in e))
