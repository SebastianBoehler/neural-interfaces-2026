"""Capture the exact hard mixer inputs and replay every downstream stage."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from patch import DOSES, EARLY, STOCK, parameters


@dataclass
class Delivery:
    x: np.ndarray
    u: np.ndarray
    base: np.ndarray
    team_b: np.ndarray
    core: np.ndarray
    extra: np.ndarray
    contrast: tuple | None
    identity: tuple | None
    recognised: bool
    team_a: np.ndarray | None
    route_out: np.ndarray
    terminal: np.ndarray
    label: int


def tap(template, signals, rows, fresh, check, device):
    model = fresh(template)
    hard = next(cls for cls in type(model).__mro__ if cls.__name__ == "HardWaveEnsemble")
    team_route = model.routeB
    mix, posterior, stepx = hard.mix6, model.team_b.posterior, team_route.stepx
    terminal_scores = getattr(model, "terminal_scores", None)
    # Some nine25 sources emit CSP scores directly via its count/CSP predict path.
    cp_step = model.cp.step
    record = {}

    def hard_mix(owner, p, members, tuned):
        q = mix(owner, p, members, tuned)
        if members:
            record["base"] = q.copy()
        return q

    def team_b(x):
        q = posterior(x)
        record["team_b"] = q.cpu().numpy().copy()
        return q

    def route(x, u, mixed, core, extra, contrast, identity):
        expected = (1 - model.TEAM_B_MIX) * record["base"] + model.TEAM_B_MIX * record["team_b"]
        np.testing.assert_array_equal(expected, mixed)
        q = stepx(x, u, mixed, core, extra, contrast, identity)
        record.update(x=x, u=u.copy(), core=core.copy(), extra=extra.copy(), contrast=contrast, identity=identity,
                      recognised=bool(model.recB), route_out=q.copy())
        return q

    def cp(*args):
        q = cp_step(*args)
        if terminal_scores is None:
            record["terminal"] = q.copy()
            record["team_a"] = None
        return q

    def terminal(q, fused):
        out = terminal_scores(q, fused)
        a = getattr(model, "team_a_p", None)
        record["team_a"] = None if a is None else a.copy()
        record["terminal"] = out.copy()
        return out

    deliveries = []
    with patch.object(hard, "mix6", hard_mix), patch.object(model.team_b, "posterior", team_b), \
            patch.object(team_route, "stepx", route), patch.object(model.cp, "step", cp):
        if terminal_scores is not None:
            model.terminal_scores = terminal
        for row in rows:
            record.clear()
            y = model.predict(signals[row:row + 1].to(device))
            if y.shape != (1,) or y.dtype != torch.long or model.dead or model.log6["err"]:
                raise ValueError(f"invalid live path at row {row}: {model.log6}")
            item = Delivery(**record, label=int(y[0]))
            if item.label != item.terminal.argmax() or not np.isfinite(item.terminal).all():
                raise ValueError(f"invalid native terminal decision at row {row}")
            deliveries.append(item)
    check(model, len(rows))
    if model.log6["v6"] != len(rows) or template.team_b.windows:
        raise ValueError("incomplete live capture or mutated session template")
    return deliveries


def replay(template, deliveries, arm, *, trace_route=False):
    route, cf, cp = deepcopy((template.routeB, template.cf, template.cp))
    if parameters(route.tail) != STOCK or route.U or cf.t or cp.t or template.v6r:
        raise ValueError(f"unexpected tape starting state: tail={parameters(route.tail)}, "
                         f"history={len(route.U)}, count={cf.t}, CSP={cp.t}, v6r={template.v6r}")
    w, schedule = DOSES.get(arm, 0.25), EARLY.get(arm, STOCK)
    terminal = getattr(type(template), "terminal_scores", None)
    state = SimpleNamespace(dead=False, TEAM_A_MIX=getattr(template, "TEAM_A_MIX", 0.0),
                            _check_probability=getattr(template, "_check_probability", None))
    labels, scores, route_scores = [], [], []
    for i, item in enumerate(deliveries):
        if i in (0, 40):
            route.tail.rb.alpha, route.tail.rb.k, route.tail.em.nmin = schedule if i == 0 else STOCK
        mixed = (1 - w) * item.base + w * item.team_b
        q = route.stepx(item.x, item.u, mixed, item.core, item.extra, item.contrast, item.identity)
        previous = cp.fused
        out = cp.step(item.x, q, cf.step(item.x, q), item.recognised)
        if terminal is not None:
            state.team_a_p = item.team_a
            out = terminal(state, out, cp.fused != previous)
        if arm == "w25":
            np.testing.assert_array_equal(q, item.route_out)
            np.testing.assert_array_equal(out, item.terminal)
        if not np.isfinite(out).all():
            raise ValueError(f"{arm} produced invalid scores on delivery {i + 1}")
        scores.append(out)
        if trace_route:
            route_scores.append(q)
        labels.append(out.argmax())
    if route.log["err"] or cf.dead or cp.dead or cf.log["err"] or cp.log["err"] or cf.t != len(deliveries):
        raise ValueError(f"{arm} did not complete with all downstream stages active")
    result = np.asarray(labels, dtype=np.int8), np.stack(scores)
    return (*result, np.stack(route_scores)) if trace_route else result
