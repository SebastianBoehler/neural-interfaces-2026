"""Capture the actual protected stream and replay private native outer states."""

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model
from settings import IDS, stamp


class Outer:
    def __init__(self, model):
        self.route, self.cf, self.cp=(deepcopy(getattr(model,k)) for k in ("routeB","cf","cp"))
        for obj in (self.route,self.cf,self.cp):
            obj.reset_state()
        assert all(x is not getattr(model,k) for x,k in zip(
            (self.route,self.cf,self.cp),("routeB","cf","cp"),strict=True))
        self.raw=[]
        self.count=[]
        self.terminal=[]
        self.fit=[]
        self.cp.refit=self.verify_refit(self.cp.refit)

    def verify_refit(self, original):
        def checked(at):
            assert len(self.raw)==len(self.cp.P)==at+1
            np.testing.assert_array_equal(self.raw[:at],self.cp.P[:at])
            self.fit.append(at)
            return original(at)
        return checked

    def step(self, event, arm):
        age=len(self.raw)
        x,u,original,alternate,core,extra,gc,gi,recognized=deepcopy(event)
        mix=alternate if arm=="hard" else original
        raw=self.route.stepx(x,u,mix,core,extra,gc,gi)
        self.raw.append(raw.copy())
        count=self.cf.step(x,raw)
        self.count.append(count.copy())
        self.terminal.append(self.cp.step(x,raw,count,recognized).copy())
        assert self.cp.t==age+1 and not self.cp.dead and not self.cf.dead
        assert not self.cp.log["err"] and not self.cf.log["err"]
        assert self.cp.stopped is None or self.cp.stopped<=age

    def result(self):
        return {k:np.asarray(getattr(self,k)) for k in ("raw","count","terminal")}


class Tap:
    def __init__(self, model, q):
        self.model,self.q=model,q
        self.pair=np.asarray(q)
        self.native=model.mix6
        self.route=model.routeB.stepx
        self.raw_frame=None
        self.consumed=[]
        self.movement=[]
        self.warm=[]
        self.weights=model.w[1+np.asarray(IDS)]
        assert np.allclose(self.weights,1/90,rtol=0,atol=1e-16)
        assert model.ft.member["fbraw"]==IDS[0] and tuple(model.ft.slots["fbraw"])==IDS
        model.routeB.stepx=self.stepping
        model.mix6=self.mixing

    def mixing(self, mix, members, warm):
        exact=self.native(mix,members,warm)
        assert members and self.raw_frame is None
        row=len(self.consumed)
        pupils=self.pair[row]
        q=pupils.mean(0)
        assert q.shape==(3,)
        np.testing.assert_allclose(q,.5*pupils[0]+.5*pupils[1],atol=1e-15,rtol=0)
        frozen=np.stack([members[i] for i in IDS])
        alternatives=dict(members)
        for i in IDS[1:]:
            alternatives[i]=q
        shifts=self.weights[1:,None]*(q-frozen[1:])
        before=stamp(warm.get("fbraw",np.zeros(3)))
        candidate=self.native(mix+shifts.sum(0),alternatives,warm)
        assert before==stamp(warm.get("fbraw",np.zeros(3)))
        expected=self.independent(frozen,q,warm,exact)
        np.testing.assert_allclose(candidate,expected,atol=2e-15,rtol=0)
        self.raw_frame=(exact,candidate)
        self.warm.append("fbraw" in warm)
        self.movement.append(float(np.abs(candidate-exact).max()))
        return exact

    def independent(self,frozen,q,warm,stock):
        native=(self.weights[:,None]*frozen).sum(0)
        substitute=self.weights[0]*frozen[0]+self.weights[1:].sum()*q
        mass=self.weights.sum()
        if "fbraw" in warm:
            move=warm["fbraw"]-frozen[0]
            def adapt(value):
                factor=np.clip(value/mass+move,1e-6,None)
                return mass*factor/factor.sum()
            native,substitute=adapt(native),adapt(substitute)
        return stock+(1-self.model.w_rest)*(substitute-native)

    def stepping(self,x,u,mix,core,extra,gc,gi):
        native,candidate=self.raw_frame
        assert native is mix
        self.raw_frame=None
        self.consumed.append(deepcopy((x,u,native,candidate,core,extra,gc,gi,self.model.recB)))
        return self.route(x,u,mix,core,extra,gc,gi)


def run_stream(template,eeg,rows,pupil,zero=False):
    model=fresh_model(template)
    compared=Outer(model)
    placebo=Outer(model) if zero else None
    prefix=Outer(model) if zero else None
    assert compared.route.tail.em is not model.routeB.tail.em and compared.cp.P is not model.cp.P
    tape=Tap(model,pupil)
    original_cp=model.cp.step
    captured={k:[] for k in ("raw","count","terminal")}

    def observe(x,raw,count,recognized):
        captured["raw"].append(raw.copy())
        captured["count"].append(count.copy())
        out=original_cp(x,raw,count,recognized)
        captured["terminal"].append(out.copy())
        return out

    model.cp.step=observe
    labels=[]
    start=perf_counter()
    for r in rows:
        out=model.predict(eeg[int(r):int(r)+1].to("cuda:0"))
        assert out.shape==(1,) and out.device.type=="cpu" and out.dtype==torch.int64
        labels.append(int(out[0]))
        assert not any(p.dead for p in (model,model.cp,model.cf))
    torch.cuda.synchronize()
    upstream_time=perf_counter()-start
    activity=replay.activity(model)
    assert activity["windows"]==len(rows)==len(tape.consumed) and tape.raw_frame is None
    assert activity["errors"]==activity["ft_over"]==0 and sum(tape.warm)==model.log6["ft"]
    np.testing.assert_array_equal(np.asarray(captured["terminal"]).argmax(1),labels)
    np.testing.assert_array_equal(np.stack([v[0] for v in tape.consumed]),eeg[rows].numpy())
    protected=stamp(np.asarray(model.ft.lab,dtype=np.int64))
    stock={k:np.stack(v) for k,v in captured.items()}
    timer=perf_counter()
    for event in tape.consumed:
        compared.step(event,"hard")
        if placebo is not None:
            placebo.step(event,"zero")
    mirror_time=perf_counter()-timer
    candidate=compared.result()
    if placebo is not None:
        for event in tape.consumed[:31]:
            prefix.step(event,"hard")
        for k,v in stock.items():
            np.testing.assert_array_equal(placebo.result()[k],v)
            np.testing.assert_array_equal(prefix.result()[k],candidate[k][:31])
    assert stamp(np.asarray(model.ft.lab,dtype=np.int64))==protected
    assert compared.cp.stopped==model.cp.stopped and max(tape.movement)>1e-6
    return stock,candidate,dict(activity=activity,candidate_refits=compared.cp.refits,candidate_fuses=compared.cp.fused,
                                candidate_fit_ages=compared.fit,stop=compared.cp.stopped,
                                protection=protected,zero_exact=zero,warm_mix=sum(tape.warm),
                                max_mix=max(tape.movement),owner_s=upstream_time,mirror_s=mirror_time)
