"""Patch and replay the three preregistered arm c extensions."""
import argparse
from functools import partial
from inspect import getsource
from pathlib import Path
from unittest.mock import patch as monkeypatch

import numpy as np

from diet_decomposition_tail import intervene as decomposition
from returning_diet import quarter_likelihood
from returning_em import build, once, returning_em_output
from tape import STOCK, replay

CANDIDATE_SHA = "cba9b208fbff7ce416efd8b5ce68ae1ad0771980d84e754433bd33e9996d0bca"
ARMS = ("em0", "rb50", "route150")


def boost_winning_ramp(contrast, identity):
    if identity is not None and identity[2] > (0 if contrast is None else contrast[2]):
        return contrast, (*identity[:2], min(1.0, identity[2] * 1.5))
    if contrast is not None and contrast[2] > 0:
        return (*contrast[:2], min(1.0, contrast[2] * 1.5)), identity
    return contrast, identity


def patch(source, arm):
    if arm == "em0":
        source = once(source, "    RECOGNISED_EM_WEIGHT = 0.25\n", "    RECOGNISED_EM_WEIGHT = 0.0\n")
    elif arm == "rb50":
        anchor = "        self.tail, self.contrast, self.identity = TailX(z), GivenGate(), GivenGate()\n"
        source = once(source, anchor, anchor + "        self.native_rb_alpha = self.tail.rb.alpha\n")
        anchor = "            out = self.routeB.stepx(x, u, mB, core1, extra, gc, gi)\n"
        source = once(source, anchor, """            self.routeB.tail.rb.alpha = self.routeB.native_rb_alpha * (0.5 if self.recB else 1.0)
""" + anchor)
    elif arm == "route150":
        anchor = "class RouteB(Route):\n"
        source = once(source, anchor, getsource(boost_winning_ramp) + "\n\n" + anchor)
        anchor = "            out = self.routeB.stepx(x, u, mB, core1, extra, gc, gi)\n"
        source = once(source, anchor, """            rc, ri = boost_winning_ramp(gc, gi) if self.recB else (gc, gi)
            out = self.routeB.stepx(x, u, mB, core1, extra, rc, ri)
""")
    else:
        raise ValueError(f"unregistered arm: {arm}")
    compile(source, "submission.py", "exec")
    return source


def intervene(model, items, arm):
    if arm == "c":
        return decomposition(model, items, "c")
    if arm not in ARMS:
        raise ValueError(f"unregistered arm: {arm}")

    tail = model.routeB.tail
    em_type, route_type = type(tail.em), type(model.routeB)
    em_step, stepx = em_type.step, route_type.stepx
    recognised = False

    def before_route(route, item):
        nonlocal recognised
        recognised = item.recognised
        if arm == "rb50":
            route.tail.rb.alpha = STOCK[0] * (0.5 if recognised else 1.0)
        return True

    def session_step(owner, x, p):
        q = em_step(owner, x, p)
        active = owner.state is not None
        if not recognised:
            return q
        if arm == "em0":
            return returning_em_output(p, q, 0.0, active)
        return quarter_likelihood(p, q, active)

    def route_step(owner, x, u, mix, core, extra, contrast, identity):
        if arm == "route150" and recognised:
            contrast, identity = boost_winning_ramp(contrast, identity)
        return stepx(owner, x, u, mix, core, extra, contrast, identity)

    with monkeypatch.object(em_type, "step", session_step), monkeypatch.object(route_type, "stepx", route_step):
        return replay(model, items, arm, trace_route=True, before_route=before_route)


def checked_replay(model, items, reference, arm):
    result = intervene(model, items, arm)
    pre_latch = np.asarray([not item.recognised for item in items])
    for actual, control in zip(result, reference, strict=True):
        np.testing.assert_array_equal(actual[pre_latch], control[pre_latch], err_msg=f"{arm} pre-latch parity")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("arm", choices=ARMS)
    args = parser.parse_args()
    build(args.parent, args.target, expected_sha=CANDIDATE_SHA, patch_source=partial(patch, arm=args.arm))


if __name__ == "__main__":
    main()
