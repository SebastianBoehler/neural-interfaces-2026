"""Build a source-only shared-recognition derivative of the signed quarter diet."""
import argparse
from inspect import getsource
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from manas_head_reachability import ZIP_SHA
from returning_em import build, once


def manas_slot(slots, contrast, identity):
    # Match the MAIN route: contrast wins a tie, and only positive ramps latch.
    gate = contrast if contrast is not None and (identity is None or contrast[2] >= identity[2]) else identity
    if gate is None or gate[2] <= 0.0 or not 0 <= gate[0] < len(slots):
        raise ValueError("MANAS did not receive a valid MAIN latch identity")
    return slots[gate[0]]


def route_slots(route_names, bank_names):
    if len(route_names) != len(set(route_names)) or set(route_names) != set(bank_names):
        raise ValueError("MAIN and MANAS enrolment names differ")
    return tuple(bank_names.index(name) for name in route_names)


def patch_main(source):
    cls = "class TerminalTeamAStack(TeamBlendStack):\n"
    text = once(source, cls, getsource(manas_slot) + "\n\n" + getsource(route_slots) + "\n\n" + cls)
    anchor = "                self._check_team_a()\n"
    text = once(text, anchor, """                names = tuple(z["route_names"].tolist())
                self.manas_slots = route_slots(names, tuple(sorted(names)))
                if (self.team_a.recipe != "A1" or len(self.team_a.bank["gallery"]) != len(names)
                        or not self.team_a.bank["enrolled"].all()):
                    raise ValueError("MANAS A1 head roster differs from MAIN")
""" + anchor)
    label = "        label = self.team_a.predict(torch.from_numpy(raw[None]), self.team_a_graph)\n"
    text = once(text, label, """        if self.recB and self.log6["rec"] == self.team_a.age:
            gate = self.route.last
            self.team_a.person = manas_slot(self.manas_slots, gate["contrast"], gate["identity"])
""" + label)
    compile(text, "submission.py", "exec")
    return text


def package(parent, candidate):
    if candidate.exists():
        raise FileExistsError(candidate)
    with TemporaryDirectory(dir=candidate.parent) as tmp:
        staged = Path(tmp) / candidate.name
        build(parent, staged, expected_sha=ZIP_SHA, patch_source=patch_main)
        os.link(staged, candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    a = parser.parse_args()
    package(a.parent, a.candidate)
