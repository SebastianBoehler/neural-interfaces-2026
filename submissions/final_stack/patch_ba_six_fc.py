"""Build the six-member MANAS candidate from the exact fused lead and a signed MANAS package."""
import argparse
from copy import copy
from hashlib import file_digest
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from build_ba_terminal import DECISION, SOLVER, TERMINAL
from patch_team_b_bank import bank_for, once, patch_bank, verify_parent
from build_cb_speed import digest

LEAD = "final_stack_hard_safe_st_team_b_nine25_tbgraph_fc"
LEAD_SHA = "7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07"
DONORS = {
    "production": "c50a62913360a7baeab978db19b778749308b171ce1d92c66343d3a8803371c5",
    "niv": "4143ac070dbedb4b4962ab734407caed4629b5607af49b5e547c8041ed65c682",
}
LICENSE = "#                                  Apache License\n"


def team_a_patch(source, donor):
    solver = "\n" + SOLVER
    if donor.count(LICENSE) != 1 or donor.count(solver) != 1 or TERMINAL not in donor:
        raise ValueError("signed MANAS donor source anchors differ")
    addition = donor[donor.index(LICENSE):donor.index(solver) + 1]
    if "    TEAM_A_MIX = 0.10\n" not in addition or "    TEAM_A_GRAPH = True\n" not in addition:
        raise ValueError("signed MANAS dose or graph differs")
    source = once(source, DECISION, TERMINAL)
    source = once(source, SOLVER, addition + SOLVER)
    source = once(source, "model = TeamBlendStack(z,", "model = TerminalTeamAStack(z,")
    compile(source, "submission.py", "exec")
    return source.encode()


def build(lead, donor, out, donor_kind):
    verify_parent(lead, LEAD_SHA)
    verify_parent(donor, DONORS[donor_kind])
    with ZipFile(lead) as parent, ZipFile(donor) as manas, TemporaryDirectory(dir=out.parent) as scratch:
        stage = Path(scratch) / out.name
        blob = bank_for("ba_six", parent.read("team_b.pt"))
        six = patch_bank(parent.read("submission.py").decode(), "ba_six", blob)
        source = team_a_patch(six, manas.read("submission.py").decode())
        payload = {"submission.py": source, "team_b.pt": blob}
        with ZipFile(stage, "x") as target:
            for entry in parent.infolist():
                target.writestr(copy(entry), payload[entry.filename] if entry.filename in payload else parent.read(entry))
            with manas.open("team_a.pt") as inp, target.open(copy(manas.getinfo("team_a.pt")), "w") as dst:
                shutil.copyfileobj(inp, dst, 2**20)
        with ZipFile(stage) as check:
            if check.namelist() != parent.namelist() + ["team_a.pt"] or check.testzip() is not None:
                raise ValueError("candidate inventory or CRC differs")
            for name in parent.namelist():
                expected = payload[name] if name in payload else parent.read(name)
                if check.read(name) != expected:
                    raise ValueError(f"unexpected change in {name}")
            with manas.open("team_a.pt") as a, check.open("team_a.pt") as b:
                if file_digest(a, "sha256").digest() != file_digest(b, "sha256").digest():
                    raise ValueError("signed MANAS weights differ")
        os.link(stage, out)
    print(digest(out), out.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lead", type=Path)
    p.add_argument("donor", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("kind", choices=tuple(DONORS))
    a = p.parse_args()
    build(a.lead, a.donor, a.out, a.kind)
