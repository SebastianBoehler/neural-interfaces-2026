"""Observe independent untouched stock upstream and complete native stage drift."""

from copy import deepcopy

import numpy as np
import torch

from same_live import native_mirrors
from smoke_gate import fresh_model



def activity(model):
    log, fit, cp, cf = model.log6, model.ft, model.cp, model.cf
    return dict(windows=log["n"], recognized=log["rec"], ft_refits=fit.refits, ft_steps=fit.steps,
                ft_relabels=fit.relabels, ft_capped=fit.capped, ft_over=fit.over, cp_refits=cp.refits,
                cp_fused=cp.fused, errors=log["err"]+cp.log["err"]+cf.log["err"])


def capture(model, windows):
    original_route, original_terminal = model.routeB.stepx, model.cp.step
    events, stages, labels, protected = [], [], [], []
    current = None

    def route(x,u,mix,core,extra,gc,gi):
        nonlocal current
        if current is not None:
            raise ValueError("stock made two RouteB calls for one raw frame")
        current = (x,u,mix,core,extra,gc,gi,model.recB)
        return original_route(x,u,mix,core,extra,gc,gi)

    def terminal(x,p,count,recognized):
        q = original_terminal(x,p,count,recognized)
        stages.append(np.stack((p,count,q)))
        return q

    model.routeB.stepx, model.cp.step = route, terminal
    try:
        for i,x in enumerate(windows):
            label = model.predict(x[None])
            if label.shape != (1,) or label.dtype != torch.long or label.device.type != "cpu":
                raise ValueError("native interface or session output changed")
            trace = model.trace
            v = trace["v6"]
            if (current is None or current[2] is not v["mix6"] or current[4] is not v["extra"]
                    or model.dead or model.cp.dead or model.cf.dead or len(stages) != i+1):
                raise ValueError("untouched stock native route or terminal path failed")
            events.append(deepcopy((*current,v["members"],v["p_ft"])))
            protected.append(dict(core=trace["core1"].copy(), core_view=trace["u"].copy(), inner=trace["mix"].copy(),
                                  shadow=trace["shadow"].copy(), v5=trace["out_v5"].copy(), outer=v["mix6"].copy(),
                                  eye=v["extra"].copy(), rest=v["rest"].copy(),
                                  members=np.stack([p for _,p in sorted(v["members"].items())]),
                                  **{f"warm_{k}": v["p_ft"].get(k,np.zeros(3)).copy() for k in model.ft.member}))
            labels.append(int(label[0]))
            current = None
    finally:
        model.routeB.stepx, model.cp.step = original_route, original_terminal
    stack = np.asarray(stages)
    np.testing.assert_array_equal(stack[:,2].argmax(1),labels)
    np.testing.assert_array_equal(np.stack([e[0] for e in events]), windows.cpu().numpy())
    arrays = {k:np.stack([d[k] for d in protected]) for k in protected[0]}
    arrays.update(raw=stack[:,0], count=stack[:,1], terminal=stack[:,2])
    return dict(events=events,stages=stack,labels=np.array(labels),protected=arrays,work=activity(model))


def effects(left,right):
    if left.keys() != right.keys():
        raise ValueError("per-stage diagnostic scopes differ")
    out = {}
    for k,a in left.items():
        b = right[k]
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid diagnostic stage {k}")
        delta = np.abs(a-b).reshape(len(a),-1).max(1)
        moved = np.flatnonzero(delta)
        out[k] = dict(max=float(delta.max()), age=int(delta.argmax()) if len(moved) else None,
                      first_changed=int(moved[0]) if len(moved) else None,
                      changed_rows=len(moved), age106=float(delta[106]) if len(delta)>106 else None)
    return out


def reference(ref, template, observed):
    stock, alt, original, alternate = native_mirrors(ref,template,observed["events"])
    np.testing.assert_array_equal(stock, observed["stages"])
    expect = observed["work"]
    if (original.cp.refits,original.cp.fused) != (expect["cp_refits"],expect["cp_fused"]):
        raise ValueError("the same-input stock mirror lost original work")
    mask = (np.arange(len(alt))>=20) & ~np.array([e[7] for e in observed["events"]])
    out = stock[:,2].copy()
    out[mask] = alt[mask,2]
    return dict(alternate_raw=alt[:,0],alternate_count=alt[:,1],alternate_terminal=alt[:,2],selected=out), \
           dict(refits=alternate.cp.refits, fuses=alternate.cp.fused, fit_ages=alternate.fit)


def aa(ref, template, windows, peer, expected, work):
    one_model = fresh_model(template)
    a,b = capture(one_model,windows), capture(fresh_model(template),windows)
    for x in (a,b):
        np.testing.assert_array_equal(x["labels"],expected)
        if x["work"] != work:
            raise ValueError("original stock activity differs from the archive")
    out_a,fit_a = reference(ref,template,a)
    out_b,fit_b = reference(ref,template,b)
    one_model.reset_state()
    one_model.predict(peer)
    one_model.reset_state()
    first = capture(one_model,windows[:1])
    reset_out,_ = reference(ref,template,first)
    if first["work"]["windows"]!=1 or one_model.cp.t!=1 or one_model.cf.t!=1:
        raise ValueError("stock active reset did not restart both clocks")
    return dict(rows=len(windows),protected=effects(a["protected"],b["protected"]),
                same_input_stock_exact=True,selected_paths=effects(out_a,out_b),alternate_work=[fit_a,fit_b],
                stock_labels_changed=int(np.count_nonzero(a["labels"]!=b["labels"])),
                selected_labels_changed=int(np.count_nonzero(out_a["selected"].argmax(1)!=out_b["selected"].argmax(1))),
                reset=effects({k:v[:1] for k,v in a["protected"].items()},first["protected"]),
                selected_reset=effects({k:v[:1] for k,v in out_a.items()},reset_out),
                reset_decision_exact=bool(np.array_equal(first["labels"],a["labels"][:1])), stock_work=work)
