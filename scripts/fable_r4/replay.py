"""Observe terminal lead scores and fit the oracle only from delivered rows."""
from time import perf_counter

import numpy as np
import torch
from scipy.special import log_softmax

from fold import D, DEVICE
from personal import CHECKPOINTS, N_VIEW, personal_fit, personal_log


def checked_predict(model, window):
    fused = model.cp.fused
    result = model.predict(window[None])
    if (result.shape != (1,) or result.dtype != torch.long or model.dead or model.log6["err"]
            or model.route.log["err"] or model.routeB.log["err"] or model.cp.dead or model.cf.dead
            or model.cp.log["err"] or model.cf.log["err"]):
        raise ValueError(f"bad model state at age {model.log6['n']}: {model.log6}")
    return int(result[0]), model.cp.fused != fused


def blend(log_lead, weight, head, pc, slow):
    return int(((1 - weight) * log_lead + weight * personal_log(head, pc, slow)).argmax())


def replay(template, windows, prefix_labels=None):
    model = D.official()[1].fresh_model(template)
    if model.team_b.windows or model.routeB.own.cur is not None:
        raise ValueError("model state crossed the session boundary")
    last = None
    step = model.cp.step

    def tap(*args):
        nonlocal last
        last = step(*args)
        return last

    model.cp.step = tap
    pc, slow, valid = [], [], []
    lead, enrolled, after100, ungated100 = [], [], [], []
    agreement, selected, seconds = [np.full(4, np.nan) for _ in range(3)]
    head = head100 = diagnostic100 = None
    weight = 0.
    active = []
    for age, raw in enumerate(torch.from_numpy(windows).to(DEVICE), 1):
        last = None
        y, fused = checked_predict(model, raw)
        if last is None or y != last.argmax() or not np.isfinite(last).all():
            raise ValueError("terminal score tap does not represent the native decision")
        if model.recB:
            head = head100 = diagnostic100 = None
        cur = model.routeB.own.cur
        lead.append(y)
        enrolled.append(y)
        after100.append(y)
        ungated100.append(y)
        if cur is not None and any(h is not None for h in (head, head100, diagnostic100)):
            lbase = log_softmax(last) if fused else np.log(np.clip(last, 1e-6, 1))
            u, s = model.trace["u"], cur[1]
            if head is not None:
                enrolled[-1] = blend(lbase, weight, head, u, s)
                active.append(age)
            if head100 is not None:
                after100[-1] = blend(lbase, .5, head100, u, s)
            if diagnostic100 is not None:
                ungated100[-1] = blend(lbase, .5, diagnostic100, u, s)

        if prefix_labels is None or model.recB or age > max(CHECKPOINTS):
            continue
        pc.append(model.trace["u"])
        valid.append(cur is not None)
        slow.append(cur[1] if cur is not None else np.zeros(N_VIEW[1]))
        if age in CHECKPOINTS:
            started = perf_counter()
            labels, confidence = prefix_labels(age)
            fitted, agree, count = personal_fit(pc, slow, valid, labels, confidence)
            k = list(CHECKPOINTS).index(age)
            agreement[k], selected[k], seconds[k] = agree, count, perf_counter() - started
            if age == 100:
                diagnostic100 = fitted
            if fitted is not None and agree >= .75:
                head, weight = fitted, CHECKPOINTS[age]
                if age == 100:
                    head100 = fitted

    if (model.log6["v6"] != len(windows) or len(model.team_b.windows) != len(windows)
            or any(g.failed for g in model.team_b.feature_graphs.values())
            or not model.team_b.feature_graphs[1].replays
            or len(windows) >= 64 and not model.team_b.feature_graphs[64].replays):
        raise ValueError("lead stages or fused Team B graphs did not process the stream")
    lead, enrolled, after100, ungated100 = (np.asarray(x, dtype=np.int8)
                                           for x in (lead, enrolled, after100, ungated100))
    recognition = model.log6["rec"] + 1
    if recognition:
        np.testing.assert_array_equal(enrolled[recognition - 1:], lead[recognition - 1:])
    np.testing.assert_array_equal(enrolled[:40], lead[:40])
    np.testing.assert_array_equal(after100[:100], lead[:100])
    np.testing.assert_array_equal(ungated100[:100], lead[:100])
    if prefix_labels is None:
        np.testing.assert_array_equal(lead, enrolled)
    return dict(lead=lead, enrolled=enrolled, after100=after100, ungated100=ungated100, loo=agreement, selected=selected,
                fit_seconds=seconds, recognition=recognition, active=np.asarray(active, dtype=np.int16))
