"""Patch only recognition-gated SessionEM in a private fold or signed ZIP."""
import argparse
from functools import partial
from inspect import getsource
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submissions/final_stack"))
from build_cb_speed import build
from patch_ba_six_fc import once

FP16_SHA = "890c2beec60344280f21b7206e99378dc71047a90b2e5db0876986b636b40cad"
ARMS = {"native": 1.0, "half": 0.5, "bypass": 0.0}


def returning_em_output(p, q, weight, active):
    if weight == 1.0 or not active:
        return q
    if weight == 0.0:
        return p
    result = np.sqrt(p * q)
    return result / result.sum()


def patch(source, weight):
    if weight not in ARMS.values():
        raise ValueError(f"unregistered returning SessionEM weight: {weight}")
    cls = "class SessionEMx(SessionEM):\n"
    text = once(source, cls, getsource(returning_em_output) + "\n\n" + cls)
    start = "    extra = None\n"
    text = once(text, start, start + """
    emission_weight = 1.0

    def step(self, x, p):
        q = super().step(x, p)
        return returning_em_output(p, q, self.emission_weight, self.state is not None)
""")
    anchor = "    def stepx(self, x, u, mix, core1, extra, contrast, identity):\n"
    text = once(text, anchor, f"    RECOGNISED_EM_WEIGHT = {weight!r}\n\n" + anchor)
    call = "            out = self.routeB.stepx(x, u, mB, core1, extra, gc, gi)\n"
    text = once(text, call, """            self.routeB.tail.em.emission_weight = (
                self.routeB.RECOGNISED_EM_WEIGHT if self.recB else 1.0
            )
""" + call)
    compile(text, "submission.py", "exec")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("arm", choices=("half", "bypass"))
    args = parser.parse_args()
    build(args.parent, args.candidate, expected_sha=FP16_SHA, patch_source=partial(patch, weight=ARMS[args.arm]))


if __name__ == "__main__":
    main()
