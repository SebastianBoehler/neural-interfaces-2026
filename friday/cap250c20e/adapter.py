"""Replace only the five final outer raw-FBC contributions before RouteB."""

from copy import deepcopy
import hashlib

import numpy as np
import torch

from settings import IDS, stamp


class RawInput:
    def __init__(self, net, owner):
        self.net, self.owner = net, owner

    def __getattr__(self, name):
        return getattr(self.net,name)

    def __call__(self, x):
        assert x.shape==(41,480) and x.dtype==np.float32
        assert self.owner.window is None
        self.owner.window=x
        return self.net(x)


class Adapter:
    def __init__(self, model, arm, templates):
        self.model, self.arm = model,arm
        assert model.routed==1 and not model.v6r and model.r15
        self.weight=model.w[1+np.asarray(IDS)]
        assert np.allclose(self.weight,1/90,rtol=0,atol=1e-16)
        assert model.ft.member["fbraw"]==IDS[0] and tuple(model.ft.slots["fbraw"])==IDS
        self.mass=float(self.weight.sum())
        self.native=model.mix6
        self.window=self.sent=None
        self.calls=0
        self.movement=[]
        self.warm=[]
        self.pupil=[]
        self.trace=[]
        self.refits=[]
        self.actors=tuple(deepcopy(a) for a in templates) if arm=="hard" else ()
        if self.actors:
            with torch.no_grad(),torch.jit.optimized_execution(False):
                assert all(a.reset().item()==0 for a in self.actors)
        self.monitor_refits()
        self.monitor_route()
        if arm!="stock":
            model.nets=RawInput(model.nets,self)
            model.mix6=self.mix6

    def monitor_refits(self):
        ft=self.model.ft
        normal=ft._refit

        def capture(age, labels, weights):
            assert age==len(labels)==len(weights) and age <= ft.cap
            assert len(ft.lab)==len(ft.wt)==age
            assert 0 < np.asarray(weights).min() <= np.asarray(weights).max() <= 1
            before=ft.steps
            prior={k:stamp(np.stack(x[:age])) for k,x in ft.X.items()}
            normal(age,labels,weights)
            assert ft.steps-before==len(ft.member)*ft.n_steps==20
            self.refits.append(dict(age=age,labels=np.asarray(labels).tolist(),weights=np.asarray(weights).tolist(),
                                    raw=prior,steps=ft.steps-before))
        ft._refit=capture

    def monitor_route(self):
        observer=self.model.routeB.stepx

        def consume(x,u,mix,core,extra,gc,gi):
            if self.arm!="stock":
                assert mix is self.sent
                self.sent=None
            self.routed=mix.copy()
            return observer(x,u,mix,core,extra,gc,gi)
        self.model.routeB.stepx=consume

    @torch.no_grad()
    def pupil_q(self, raw):
        x=torch.from_numpy(np.ascontiguousarray(raw[None])).to("cuda:0")
        with torch.jit.optimized_execution(False):
            p=[a(x) for a in self.actors]
        assert len(p)==2 and all(q.shape==(1,3) and q.dtype==torch.float64 for q in p)
        pair=np.stack([q.cpu().numpy()[0] for q in p])
        assert np.isfinite(pair).all() and (pair>=0).all()
        np.testing.assert_allclose(pair.sum(1),1.,rtol=0,atol=1e-12)
        self.pupil.append(pair)
        return pair.mean(0)

    def mix6(self, mix, members, p_ft):
        if not members:
            return self.native(mix,members,p_ft)
        assert self.window is not None and self.sent is None
        base=np.stack([members[i] for i in IDS])
        old_family=(self.weight[:,None]*base).sum(0)
        native_mix=self.native(mix,members,p_ft)
        altered=dict(members)
        if self.actors:
            mean=self.pupil_q(self.window)
            for idx in IDS[1:]:
                altered[idx]=mean
        new=np.stack([altered[i] for i in IDS])
        delta=(self.weight[:,None]*(new-base)).sum(0)
        new_family=(self.weight[:,None]*new).sum(0)
        if self.actors:
            np.testing.assert_allclose(new_family/self.mass,(base[0]+5*mean)/6.,rtol=0,atol=8e-16)
        out=self.native(mix+delta,altered,p_ft)
        assert all(altered[i] is members[i] for i in members if i not in IDS[1:])
        if "fbraw" in p_ft:
            shift=p_ft["fbraw"]-base[0]
            ref=np.clip(old_family/self.mass+shift,1e-6,None)
            got=np.clip(new_family/self.mass+shift,1e-6,None)
            ref=ref*self.mass/ref.sum()
            got=got*self.mass/got.sum()
        else:
            ref,got=old_family,new_family
        predicted=native_mix+(1-self.model.w_rest)*(got-ref)
        np.testing.assert_allclose(out,predicted,rtol=0,atol=2e-15)
        if self.arm=="identity":
            np.testing.assert_array_equal(out,native_mix)
        assert np.isfinite(out).all() and float(out.min())>=0 and abs(float(out.sum())-1)<=1e-12
        self.movement.append(float(np.abs((got-ref)/self.mass).max()))
        self.warm.append("fbraw" in p_ft)
        self.calls+=1
        self.sent=out
        self.window=None
        return out

    def record(self):
        t=self.model.trace
        now=t["v6"]
        np.testing.assert_array_equal(self.routed,now["mix6"])
        if self.arm!="stock":
            assert self.calls==len(self.trace)+1 and self.window is self.sent is None
        vals = [t[k] for k in ("mix","shadow","out_v5")]
        vals.append(now["label"])
        for fam in sorted(self.model.ft.member):
            vals.append(now["p_ft"].get(fam,np.zeros(3)))
        self.trace.append(np.stack(vals))

    def summary(self):
        value=np.asarray(self.trace)
        assert value.shape==(len(self.trace),8,3)
        return dict(flavor=self.arm,raw_calls=self.calls,moved=sum(x>.01 for x in self.movement),
                    max_move=max(self.movement,default=0.),warm_rows=sum(self.warm),
                    refit_ages=[r["age"] for r in self.refits], upstream=stamp(value),
                    optimizer_prior=hashlib.sha256(repr(self.refits).encode()).hexdigest())
