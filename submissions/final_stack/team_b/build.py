"""Build and verify new hardcb_safe Team B fold copies or a final ZIP."""
import argparse
import hashlib
import io
import os
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import sys

sys.path.insert(0, str(HERE.parent))
from patch_dead import NEW, OLD

SAFE_SHA = "8a868f22233f73bc9748003667729f4eaa4b31e3e0ae4ca880b1b71bc64480f7"
FACTORY = "model = StackEnsemble(z,"
SOLVER = "class Solver(CompetSolver):"
ARMS = {"base": (0., 3), "nine15": (0.15, 3), "nine25": (0.25, 3),
        "three15": (0.15, 1), "three25": (0.25, 1)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_source(original):
    if original.count(OLD) == 1 and NEW not in original:
        original = original.replace(OLD, NEW)
    if original.count(NEW) != 1 or OLD in original:
        raise ValueError("invalid hardcb dead-session guard")
    return original


def encoder_source():
    original = (ROOT / "cleanslate/teamB/submission.py").read_text()
    start, end = original.index("FS, T, C, PD ="), original.index("def balance(")
    return ("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n" +
            original[start:end] + (HERE / "encoder.py.inc").read_text())


def bank(bank_path, members):
    pack = torch.load(bank_path, map_location="cpu", weights_only=True)
    nets = pack["nets"]
    if tuple(k for k, _ in nets) != ("shallow",) * 3 + ("slow",) * 3 + ("fbc",) * 3:
        raise ValueError("expected three seeds of each R1 family")
    if pack.get("cfg") or pack.get("experts") or pack.get("ext") or pack["steps"] != 1200 or pack["lam"] != 0.5:
        raise ValueError("expected the registered R1 recipe")
    print(f"Team B source SHA-256 {sha(Path(bank_path).read_bytes())}", flush=True)
    pack["nets"] = [nets[i + j] for i in (0, 3, 6) for j in range(members)]
    out = io.BytesIO()
    torch.save(pack, out)
    return out.getvalue()


def revised(original, weight, members, blob):
    original = safe_source(original)
    if original.count(FACTORY) != 1 or original.count(SOLVER) != 1:
        raise ValueError("expected one hardcb Solver factory")
    encoder = encoder_source()
    compile(encoder, "<Team B encoder>", "exec")
    stack = (HERE / "stack.py.inc").read_text()
    stack = stack.replace("__BLEND__", repr(weight)).replace("__MEMBERS__", repr(members))
    stack = stack.replace("__SHA__", repr(sha(blob)))
    inserted = f"_TEAM_B = {{}}\nexec(compile({encoder!r}, '<Team B encoder>', 'exec'), _TEAM_B)\n\n"
    src = original.replace(SOLVER, inserted + stack + "\n\n" + SOLVER)
    src = src.replace(FACTORY, FACTORY.replace("StackEnsemble", "TeamBlendStack"))
    compile(src, "submission.py", "exec")
    return src


def fold(src, out, weight, members, blob):
    out.mkdir(parents=True)
    (out / "submission.py").write_text(revised((src / "submission.py").read_text(), weight, members, blob))
    for path in src.iterdir():
        if path.is_file() and path.name != "submission.py":
            os.link(path, out / path.name)
    (out / "team_b.pt").write_bytes(blob)


def archive(src, out, arm, bank_path):
    if sha(src.read_bytes()) != SAFE_SHA:
        raise ValueError("hardcb_safe archive SHA-256 differs")
    weight, members = ARMS[arm]
    blob = bank(bank_path, members)
    with TemporaryDirectory(dir=out.parent) as tmp, ZipFile(src) as original:
        stage = Path(tmp) / "staged.zip"
        with ZipFile(stage, "x", compression=ZIP_DEFLATED) as pending:
            for entry in original.infolist():
                data = original.read(entry)
                if entry.filename == "submission.py":
                    data = revised(data.decode(), weight, members, blob).encode()
                pending.writestr(copy(entry), data)
            pending.writestr("team_b.pt", blob)
        with ZipFile(stage) as check:
            assert check.namelist() == original.namelist() + ["team_b.pt"] and check.testzip() is None
            assert all(check.read(n) == original.read(n) for n in original.namelist() if n != "submission.py")
        os.link(stage, out)
    print(sha(out.read_bytes()), out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("arm", choices=tuple(ARMS)[1:])
    p.add_argument("bank", type=Path)
    a = p.parse_args()
    archive(a.src, a.out, a.arm, a.bank)
