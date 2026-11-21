"""Patch the signed em_half archive with the registered returning adaptation diet."""
import argparse
from functools import partial
from inspect import getsource
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from returning_em import build, once
from build_cb_speed import digest

PARENT_SHA = "833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d"
ARMS = {"zero": 0.0, "quarter": 0.25}


def quarter_likelihood(p, q, active):
    if not active:
        return q
    out = np.sqrt(p * np.sqrt(p * q))
    return out / out.sum()


def diet_tail_step(tail, x, u, mix):
    p = tail.rb.step(mix)
    if tail.diet_weight == 0.0:
        return p
    q = tail.em.step(x, p)
    q = quarter_likelihood(p, q, tail.em.state is not None)
    out = tail.tsem.step(u, q, tail.em.gate)
    return quarter_likelihood(q, out, tail.tsem.state is not None)


def patch(source, weight):
    if weight not in ARMS.values():
        raise ValueError(f"unregistered diet weight: {weight}")
    anchor = "class TailX(Tail):\n"
    helpers = getsource(quarter_likelihood) + "\n\n" + getsource(diet_tail_step) + "\n\n"
    source = once(source, anchor, helpers + anchor + """    diet_weight = None

    def reset_state(self):
        super().reset_state()
        self.diet_weight = None

""")
    stock_step = "        return super().step(x, u, mix)\n"
    source = once(source, stock_step, """        if self.diet_weight is not None:
            return diet_tail_step(self, x, u, mix)
""" + stock_step)
    source = once(source, "    RECOGNISED_EM_WEIGHT = 0.5\n", f"    RECOGNISED_DIET_WEIGHT = {weight!r}\n")
    source = once(source, """            self.routeB.tail.em.emission_weight = (
                self.routeB.RECOGNISED_EM_WEIGHT if self.recB else 1.0
            )
""", """            self.routeB.tail.diet_weight = self.routeB.RECOGNISED_DIET_WEIGHT if self.recB else None
""")
    stock_count = "                q = self.cp.step(x, p, self.cf.step(x, p), self.recB if self.routed else not self.active)\n"
    source = once(source, stock_count, """                recognised = self.recB if self.routed else not self.active
                count = p if self.recB else self.cf.step(x, p)
                q = self.cp.step(x, p, count, recognised)
""")
    compile(source, "submission.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("arm", choices=ARMS)
    args = parser.parse_args()
    if args.candidate.exists():
        raise FileExistsError(args.candidate)
    with TemporaryDirectory(dir=args.candidate.parent) as temp:
        staged = Path(temp) / args.candidate.name
        build(args.parent, staged, expected_sha=PARENT_SHA, patch_source=partial(patch, weight=ARMS[args.arm]))
        os.link(staged, args.candidate)
    print(digest(args.candidate), args.candidate, flush=True)


if __name__ == "__main__":
    main()
