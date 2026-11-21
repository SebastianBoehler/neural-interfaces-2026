"""Apply the component changes to the full Fable R6 causal tail."""
from unittest.mock import patch as monkeypatch

from diet_decomposition_patch import ARMS, QUARTER, quarter_likelihood
from returning_em import returning_em_output
from tape import replay

SETTINGS = {"all": QUARTER, "half": (False, False, False), **ARMS}


def intervene(model, items, arm):
    count, tsem, em = SETTINGS[arm]
    tail = model.routeB.tail
    em_step = type(tail.em).step
    tsem_step = type(tail.tsem).step
    recognised = False

    def latch(_route, item):
        nonlocal recognised
        recognised = item.recognised
        return not (count and recognised)

    def session_step(owner, x, p):
        q = em_step(owner, x, p)
        if not recognised:
            return q
        active = owner.state is not None
        return quarter_likelihood(p, q, active) if em else returning_em_output(p, q, 0.5, active)

    def template_step(owner, u, q, gate):
        out = tsem_step(owner, u, q, gate)
        return quarter_likelihood(q, out, owner.state is not None) if tsem and recognised else out

    with monkeypatch.object(type(tail.em), "step", session_step), monkeypatch.object(type(tail.tsem), "step", template_step):
        return replay(model, items, arm, trace_route=True, before_route=latch)
