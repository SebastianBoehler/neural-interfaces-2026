"""Observe the actual packaged waveform call and independently reconstruct native tails."""

from copy import deepcopy

import numpy as np

from adapter import Adapter, RawInput
from benchmark_utils.streaming import fresh_model
from mirror import Outer
from stock_drift import capture


EPS = 2e-15


class ObserveHard:
    def __init__(self, model, expected):
        self.model, self.expected = model, expected
        self.window = None
        self.inputs, self.pairs = [], []
        self.original_pupil, self.original_mix = model.hard_probability, model.mix6
        model.nets = RawInput(model.nets,self)
        model.hard_probability, model.mix6 = self.pupil, self.mix

    def pupil(self, raw):
        if raw is not self.window or raw is not self.model.hard_window:
            raise ValueError("the CPU pupil did not receive the exact original NetBank frame")
        q = self.original_pupil(raw)
        np.testing.assert_array_equal(q,self.expected[len(self.pairs)])
        if len(self.model.hard_actors) != 2 or any(p.device.type != "cpu" for a in self.model.hard_actors for p in a.parameters()):
            raise ValueError("packaged hard actors must run on CPU")
        self.pairs.append(q.copy())
        self.window = None
        return q

    def mix(self, mix, members, warm):
        if not members:
            raise ValueError("cap250 routed native path did not produce its complete member set")
        prev = deepcopy((mix,members,warm,self.model.rest_p))
        result = self.original_mix(mix,members,warm)
        if self.window is not None or len(self.pairs) != len(self.inputs)+1:
            raise ValueError("the actual package mixer bypassed its signed actors")
        self.inputs.append(prev)
        return result


def proof(stock, live, seen):
    baseline, candidate = (Outer(stock) for _ in range(2))
    native = fresh_model(stock)
    ids = tuple(range(47,53))
    weights = native.w[1+np.asarray(ids)]
    scale = weights.sum()
    motion, warm, chosen = [], 0, []
    for event,payload,pair in zip(live["events"],seen.inputs,seen.pairs,strict=True):
        x,u,pack,core,extra,gc,gi,recognized,raw,actual_warm = event
        mix,members,p_ft,rest = payload
        for old,new in ((members,raw),(p_ft,actual_warm)):
            if old.keys() != new.keys():
                raise ValueError("actual native member or warm family inventory changed")
            for k in old:
                np.testing.assert_array_equal(old[k],new[k])
        native.rest_p = rest
        origin = native.mix6(mix,members,p_ft)
        frozen = np.stack([members[i] for i in ids])
        mean = pair.mean(0)
        altered = dict(members)
        altered.update({i:mean for i in ids[1:]})
        d = (weights[1:,None]*(mean-frozen[1:])).sum(0)
        alternate = native.mix6(mix+d,altered,p_ft)
        np.testing.assert_array_equal(alternate,pack)
        before = (weights[:,None]*frozen).sum(0)/scale
        after = (frozen[0]+5*mean)/6
        if "fbraw" in p_ft:
            warm += 1
            shift=p_ft["fbraw"]-frozen[0]
            before,after=(np.clip(p+shift,1e-6,None) for p in (before,after))
            before,after=before/before.sum(),after/after.sum()
        np.testing.assert_allclose(alternate,origin+(1-native.w_rest)*scale*(after-before),rtol=0,atol=EPS)
        motion.append(float(np.abs(after-before).max()))
        baseline.step((x,u,origin,origin,core,extra,gc,gi,recognized),"stock")
        e=(x,u,origin,alternate,core,extra,gc,gi,recognized)
        candidate.step(e,"hard")
        chosen.append(e)
    np.testing.assert_array_equal(np.stack((candidate.raw,candidate.count,candidate.terminal),axis=1),live["stages"])
    short=Outer(stock)
    n=min(31,len(chosen))
    for event in chosen[:n]:
        short.step(event,"hard")
    np.testing.assert_array_equal(np.stack((short.raw,short.count,short.terminal),axis=1),live["stages"][:n])
    if candidate.cp.fused!=live["work"]["cp_fused"] or candidate.cp.refits!=live["work"]["cp_refits"]:
        raise ValueError("actual packaged CSP work differs from its private same-input actor")
    if max(motion)<=.01:
        raise ValueError("native five-of-six raw-family replacement is inactive")
    return dict(calls=len(chosen),warm_rows=warm,max_family_delta=max(motion),sha_pupils_exact=True,
                exact_stock_and_candidate_native_inputs=True,exact_private_tail=True,exact_same_input_prefix=True,
                same_live_terminal_delta=float(np.abs(np.asarray(baseline.terminal)-live["stages"][:,2]).max()),
                csp_ages=candidate.fit,csp_fusions=candidate.cp.fused)


def measured(template, windows, expected=None):
    model = fresh_model(template)
    monitor = Adapter(model,"stock",())
    tap = ObserveHard(model,expected) if expected is not None else None
    result = capture(model,windows)
    return model,result,monitor.refits,tap


def protect(stock, trial, tolerance=5e-5):
    prior, candidate = stock[1],trial[1]
    fields = prior["work"].keys()-{"cp_refits","cp_fused"}
    if any(prior["work"][k]!=candidate["work"][k] for k in fields):
        raise ValueError("protected native recognition/refit work changed")
    prior_weight_max=gate_score_max=0.
    for a,b in zip(stock[2],trial[2],strict=True):
        if any(a[k]!=b[k] for k in ("age","raw","labels","steps")):
            raise ValueError("optimizer consumed different strict-prior rows or labels")
        d=float(np.abs(np.asarray(a["weights"])-b["weights"]).max())
        prior_weight_max=max(prior_weight_max,d)
        if tolerance is not None and d>tolerance:
            raise ValueError("independent prior confidence changed beyond the fixed bound")
    for old,new in zip(prior["events"],candidate["events"],strict=True):
        if not np.array_equal(old[0],new[0]) or old[7] != new[7]:
            raise ValueError("actual delivered frame or recognition state differs")
        for a,b in zip(old[5:7],new[5:7],strict=True):
            if (a is None) != (b is None):
                raise ValueError("native recognition gate presence changed")
            if a is not None:
                if a[0]!=b[0] or a[2]!=b[2]:
                    raise ValueError("native recognition route or ramp changed")
                d=abs(float(a[1])-float(b[1]))
                gate_score_max=max(gate_score_max,d)
                if tolerance is not None and d>tolerance:
                    raise ValueError("independent native gate score changed beyond the fixed bound")
    drift={}
    for k,a in prior["protected"].items():
        if k not in ("outer","raw","count","terminal"):
            b=candidate["protected"][k]
            if tolerance is not None:
                np.testing.assert_allclose(a,b,rtol=0,atol=tolerance,err_msg=k)
            if k in ("inner","shadow","v5"):
                np.testing.assert_array_equal(a.argmax(1),b.argmax(1))
            delta=np.abs(a-b).reshape(len(a),-1).max(1)
            active=np.flatnonzero(delta)
            drift[k]=dict(max=float(delta.max()),age=int(delta.argmax()) if len(active) else None,
                          first=int(active[0]) if len(active) else None,age106=float(delta[106]) if len(a)>106 else None)
    return dict(stages=drift,prior_weight_max=prior_weight_max,gate_score_max=gate_score_max)
