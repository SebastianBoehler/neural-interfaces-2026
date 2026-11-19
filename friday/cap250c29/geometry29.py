"""Insert a transported prior in the real native CSP eigensolver."""
from copy import deepcopy
import inspect
from textwrap import dedent
from types import MethodType

import numpy as np

from files29 import EXTERNAL_ARMS,stamp


MARK="                Ma, Mo = np.average(Cb[a], 0, np.ones(a.sum())), np.average(Cb[o], 0, np.ones(o.sum()))\n"
PRIOR_ROWS=12


def symmetric(a):
    return .5*(a+np.swapaxes(a,-1,-2))


def recolor(past,source):
    base=symmetric(np.mean(past,axis=0))
    s,u=np.linalg.eigh(base)
    assert s.min()>0 and base.shape==(8,41,41)
    R=(u*s[:,None,:]**.5)@u.transpose(0,2,1)
    signs=np.where(np.arange(41)[::-1]%2,-1.,1.)
    Q=(u*signs)@u.transpose(0,2,1)
    oriented=symmetric(R[:,None]@source@R[:,None])
    reflection=symmetric(Q[:,None]@oriented@Q[:,None])
    np.testing.assert_allclose(Q@Q,np.broadcast_to(np.eye(41),Q.shape),atol=2e-12,rtol=0)
    np.testing.assert_allclose(reflection.mean(1),base,atol=2e-9,rtol=2e-11)
    np.testing.assert_allclose(np.linalg.eigvalsh(reflection),np.linalg.eigvalsh(oriented),atol=2e-9,rtol=2e-11)
    np.testing.assert_allclose(np.trace(reflection,axis1=-2,axis2=-1),np.trace(oriented,axis1=-2,axis2=-1),rtol=2e-11)
    physical=float((np.linalg.norm(oriented-reflection,axis=(-2,-1))/np.linalg.norm(oriented,axis=(-2,-1))).max())
    return {"transport":oriented,"reflection":reflection,"pooled":np.repeat(base[:,None],3,axis=1)},physical


def native_hook(bound):
    raw=dedent(inspect.getsource(bound.__func__))
    anchor=dedent(MARK).strip()
    assert raw.count(anchor)==1 and raw.count("eigh(Ma, Ma + Mo)")==1
    edit=anchor+"\n            Ma, Mo = self.prior29(t, b, y[tr], cl, Ma, Mo)"
    derived=raw.replace(anchor,edit)
    namespace=dict(bound.__func__.__globals__)
    exec(compile(derived,"<F-C29 native CausalPers.refit>","exec"),namespace)
    return MethodType(namespace["refit"],bound.__self__)


def shrink(native,prior,n):
    w=PRIOR_ROWS/(PRIOR_ROWS+n)
    return (1-w)*native+w*prior*(np.trace(native)/np.trace(prior))


class Spatial:
    def __init__(self,template,source,arm):
        assert arm in EXTERNAL_ARMS
        self.cp=deepcopy(template)
        self.cp.reset_state()
        assert self.cp.t==0 and not self.cp.C and self.cp.model is None
        self.source,self.arm=source,arm
        self.fit=self.cp.refit if arm=="stock" else native_hook(self.cp.refit)
        self.cp.refit=self.monitor
        self.cp.prior29=self.mix
        self.events=[]
        self.selected,self.record,self.cache=None,None,None

    def mix(self,t,band,labels,cl,a,o):
        assert self.selected is not None and self.record is not None
        assert self.record["age"]==t and t>self.selected[-1]
        self.record["prior_calls"]+=1
        if self.arm=="identity":
            return a,o
        if self.cache is None:
            all_priors,physical=recolor(np.stack(self.cp.C[:t]),self.source)
            self.cache=all_priors[self.arm]
            self.record["reflection_delta"]=physical
        p=self.cache[band]
        members=np.bincount(labels,minlength=3)
        n=int(members[cl])
        members[cl]=0
        no=int(members.sum())
        assert n>0 and no>0
        p_o=np.average(p,axis=0,weights=members)
        ma,mo=shrink(a,p[cl],n),shrink(o,p_o,no)
        rel=max(np.linalg.norm(ma-a)/np.linalg.norm(a),np.linalg.norm(mo-o)/np.linalg.norm(o))
        self.record["mean_move"]=max(self.record["mean_move"],float(rel))
        return ma,mo

    def monitor(self,t):
        cp=self.cp
        assert len(cp.C)==len(cp.P)==t+1 and cp.t==t+1
        native=np.asarray(cp.P[:t]); y=native.argmax(1)
        self.record=dict(age=t,prior_calls=0,mean_move=0.,reflection_delta=0.)
        select=cp.select

        def observed(labels,conf):
            assert self.selected is None
            np.testing.assert_array_equal(labels,y)
            np.testing.assert_array_equal(conf,native.max(1))
            out=select(labels,conf)
            assert len(out)>0 and max(out)<t
            self.selected=out
            return out

        cp.select=observed
        try:
            result=self.fit(t)
        finally:
            cp.select=select
        use=self.selected
        assert use is not None
        n=len(set(y[use]))
        assert (result is not None)==(n>=2)
        if result is None:
            bands,width,classes=[],0,[]
        else:
            filters,head=result
            bands=[stamp(w) for w in filters]
            assert all(w.shape==(41,4*n) for w in filters)
            width=head.n_features_in_
            classes=head.classes_.tolist()
            assert width==8*4*n and classes==sorted(set(y[use]))
        expect=8*n if result is not None and self.arm!="stock" else 0
        assert self.record["prior_calls"]==expect
        self.events.append(dict(**self.record,selected=stamp(use),n=len(use),label=stamp(y[use]),
                                fits=bool(result is not None),width=int(width),head=classes,basis=bands,
                                classes=np.bincount(y[use],minlength=3).tolist()))
        self.selected=self.record=self.cache=None
        return result

    def run(self,x,raw,count,rec):
        act=self.cp
        out=np.stack([act.step(a,b,c,d) for a,b,c,d in zip(x,raw,count,rec,strict=True)])
        assert not act.dead and not act.log["err"] and act.t==len(x) and len(self.events)==act.refits
        stop=None if not np.any(rec) else int(np.flatnonzero(rec)[0])
        assert act.stopped==stop
        last=len(x) if stop is None else stop
        assert [v["age"] for v in self.events]==list(range(act.t0,last,act.k))
        assert act.fused==max(0,last-act.w0)
        np.testing.assert_array_equal(out[:act.w0],count[:act.w0])
        if stop is not None:
            np.testing.assert_array_equal(out[stop:],count[stop:])
        return out,dict(refits=act.refits,fusions=act.fused,stop=stop,fits=self.events,
                weight=float(act.cap),max_mean_move=max((p["mean_move"] for p in self.events),default=0.),
                max_reflection=max((p["reflection_delta"] for p in self.events),default=0.))


def matched(cp,source,x,raw,count,rec,expected,arms=EXTERNAL_ARMS):
    fits={a:Spatial(cp,source,a).run(x,raw,count,rec) for a in arms}
    stock=fits["stock"]
    np.testing.assert_array_equal(stock[0],expected)
    if "identity" in fits:
        np.testing.assert_array_equal(fits["identity"][0],expected)
    for a,(out,proof) in fits.items():
        assert all(proof[k]==stock[1][k] for k in ("refits","fusions","stop","weight"))
        for ref,p in zip(stock[1]["fits"],proof["fits"],strict=True):
            assert all(p[k]==ref[k] for k in ("age","selected","n","label","classes","fits","width","head"))
            if a=="identity":
                assert p["basis"]==ref["basis"]
        proof["score_move"]=float(np.abs(out-expected).max())
        if a in ("transport","reflection","pooled") and proof["fusions"]:
            assert proof["max_mean_move"]>1e-4 and proof["max_reflection"]>1e-3
            assert any(v["basis"]!=w["basis"] for v,w in zip(proof["fits"],stock[1]["fits"],strict=True))
            assert proof["score_move"]>1e-4
    if "reflection" in fits and stock[1]["fusions"]:
        assert float(np.abs(fits["transport"][0]-fits["reflection"][0]).max())>1e-4
    return np.stack([fits[a][0] for a in arms]),{a:fits[a][1] for a in arms}
