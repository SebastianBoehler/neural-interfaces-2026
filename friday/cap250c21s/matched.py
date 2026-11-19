"""Run genuine independent native outer states on one live cap250 upstream."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

import replay
from adapter import Adapter as RefitAudit
from benchmark_utils.streaming import fresh_model
from fbl2 import FBL2Hook
from mirror import Outer
from route_probe import RouteProbe
from settings import stamp


def capture(template, eeg, rows):
    model=fresh_model(template)
    assert not model.v5hist and not model.cp.t and not model.dead
    refit=RefitAudit(model,"stock",())
    route, terminal=model.routeB.stepx, model.cp.step
    tape, emissions=[],[]
    active=None

    def record_input(x,u,mix,core,extra,gc,gi):
        nonlocal active
        assert active is None
        active=(x,u,mix,core,extra,gc,gi,model.recB)
        return route(x,u,mix,core,extra,gc,gi)

    def record_terminal(x,raw,count,recognized):
        out=terminal(x,raw,count,recognized)
        emissions.append(np.stack((raw,count,out)))
        return out

    model.routeB.stepx,model.cp.step=record_input,record_terminal
    labels=[]
    for i in rows:
        y=model.predict(eeg[int(i):int(i)+1].to("cuda:0"))
        assert y.shape==(1,) and y.dtype==torch.int64 and y.device.type=="cpu"
        assert not any(p.dead for p in (model,model.cf,model.cp))
        t=model.trace["v6"]
        assert active[2] is t["mix6"] and active[4] is t["extra"]
        tape.append(deepcopy((*active,t["members"],t["p_ft"])))
        labels.append(int(y[0]))
        active=None
        refit.record()
    stock=np.asarray(emissions)
    np.testing.assert_array_equal(stock[:,2].argmax(1),labels)
    np.testing.assert_array_equal(np.stack([r[0] for r in tape]),eeg[rows].numpy())
    activity=replay.activity(model)
    assert len(rows)==len(tape)==activity["windows"] and activity["ft_over"]==activity["errors"]==0
    assert not model.dead and not model.cf.dead and not model.cp.dead
    assert activity["ft_steps"]==20*activity["ft_refits"]
    assert model.cp.stopped==(activity["recognized"] if activity["recognized"]>=0 else None)
    assert len(refit.refits)==activity["ft_refits"]
    assert sum("fbl" in r[-1] for r in tape)==model.log6["ft"]
    protected=dict(trace=stamp(np.asarray(refit.trace)), inputs=stamp(np.stack([r[2] for r in tape])),
                   labels=stamp(np.asarray(model.ft.lab)), activity=activity,
                   prior_rows=sum(a["age"] for a in refit.refits), fit_ages=[a["age"] for a in refit.refits],
                   stop=model.cp.stopped, stock_fits=model.cp.refits, stock_fuses=model.cp.fused)
    return tape,stock,protected


def private(outer, tape, mix):
    assert not outer.raw and not outer.cp.t and not outer.route.U
    for (x,u,_,core,extra,gc,gi,recognized,_,_), p in zip(tape,mix,strict=True):
        # Set both mixture positions to the chosen current input. Outer owns native RouteB, CF and CSP.
        outer.step((x,u,p,p,core,extra,gc,gi,recognized),"stock")
    return np.stack((outer.raw, outer.count, outer.terminal),axis=1)


def paired(template,eeg,rows, *, prefix=False, expected_stock=None):
    tape,stock,protected=capture(template,eeg,rows)
    live_mix=[r[2] for r in tape]
    assert protected["inputs"]==stamp(np.asarray(live_mix))
    baseline=Outer(template)
    zero=private(baseline,tape,live_mix)
    np.testing.assert_array_equal(zero,stock)
    assert baseline.cp.refits==protected["stock_fits"] and baseline.cp.fused==protected["stock_fuses"]
    assert baseline.cp.stopped==protected["stop"]
    base_y=stock[:,2].argmax(1).tolist()
    if expected_stock is not None:
        assert base_y==expected_stock

    # Allocate the independent FBL2 hook only after the complete live stock session.
    candidate_solver=fresh_model(template)
    hook=FBL2Hook(candidate_solver,"fbl2")
    candidate=Outer(candidate_solver)
    consuming=RouteProbe(SimpleNamespace(routeB=candidate.route))
    assert candidate.route.a==1/17 and baseline.route.a==hook.original_share
    assert candidate.route.tail.em is not baseline.route.tail.em and candidate.cp.P is not baseline.cp.P
    boosted=[hook.project(r[2],r[-2],r[-1]) for r in tape]
    trial=private(candidate,tape,boosted)
    assert protected["inputs"]==stamp(np.asarray(live_mix))
    assert candidate.cp.stopped==baseline.cp.stopped and hook.warm==sum("fbl" in r[-1] for r in tape)
    assert hook.moved==len(rows) and hook.mix_delta>.001
    assert np.isfinite(trial).all() and float(np.abs(trial-stock).max())>1e-4
    if prefix:
        # Use a different captured visit elsewhere in the smoke before repeating this branch prefix.
        short=Outer(candidate_solver)
        np.testing.assert_array_equal(private(short,tape[:31],boosted[:31]),trial[:31])
    route=consuming.summary()
    assert route["routed"]==candidate.route.log["routed"]
    result=dict(stock=base_y,fbl2=trial[:,2].argmax(1).tolist(),rows=rows.tolist(),
                identity_exact=True, prefix_exact=prefix, protected=protected, hook=hook.summary(),
                route=route, trial_fits=candidate.fit, trial_fuses=candidate.cp.fused,
                terminal_change=float(np.abs(trial[:,2]-stock[:,2]).max()))
    return result
