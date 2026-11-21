"""Patch individual recognition-gated corrections on signed em_half source."""
import argparse
from functools import partial
from inspect import getsource
from pathlib import Path

from returning_diet import PARENT_SHA, quarter_likelihood
from returning_em import build, once
from build_cb_speed import digest

ARMS = {
    "a": (True, False, False),
    "b": (False, True, False),
    "c": (False, False, True),
    "d": (True, True, False),
}
QUARTER = (True, True, True)
STOCK_COUNT = "                q = self.cp.step(x, p, self.cf.step(x, p), self.recB if self.routed else not self.active)\n"
STOCK_TAIL = "        return super().step(x, u, mix)\n"


def patch(source, settings):
    count, tsem, em = settings
    if tsem or em:
        anchor = "def returning_em_output(p, q, weight, active):\n"
        source = once(source, anchor, getsource(quarter_likelihood) + "\n\n" + anchor)
    if em:
        source = once(source, "    RECOGNISED_EM_WEIGHT = 0.5\n", "    RECOGNISED_EM_WEIGHT = 0.25\n")
        source = once(source, "    if weight == 0.0:\n        return p\n", """    if weight == 0.0:
        return p
    if weight == 0.25:
        return quarter_likelihood(p, q, active)
""")
    if tsem:
        source = once(source, "class TailX(Tail):\n", """class TailX(Tail):
    tsem_quarter = False

    def reset_state(self):
        super().reset_state()
        self.tsem_quarter = False

""")
        source = once(source, STOCK_TAIL, """        if self.tsem_quarter:
            q = self.em.step(x, self.rb.step(mix))
            out = self.tsem.step(u, q, self.em.gate)
            return quarter_likelihood(q, out, self.tsem.state is not None)
""" + STOCK_TAIL)
        anchor = "            out = self.routeB.stepx(x, u, mB, core1, extra, gc, gi)\n"
        source = once(source, anchor, "            self.routeB.tail.tsem_quarter = self.recB\n" + anchor)
    if count:
        source = once(source, STOCK_COUNT, """                recognised = self.recB if self.routed else not self.active
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
    build(args.parent, args.candidate, expected_sha=PARENT_SHA, patch_source=partial(patch, settings=ARMS[args.arm]))
    print(digest(args.candidate), args.candidate, flush=True)


if __name__ == "__main__":
    main()
