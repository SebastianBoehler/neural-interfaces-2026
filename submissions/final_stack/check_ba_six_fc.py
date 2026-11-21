"""Exercise the exact fused six-member archive, the six-member control, and hard zero."""
import argparse
from contextlib import ExitStack
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

import check_ba_terminal as G
from build_hard_team_b import BASE, BASE_SHA
from patch_ba_six_fc import LEAD, LEAD_SHA
from patch_team_b_bank import patch_bank

C = G.C


def load_folder(folder, data, device):
    spec = spec_from_file_location("ba_six_control", folder / "submission.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return C.load(mod, folder, data, device)


def zero(zips, module, folder, data, scratch):
    lead_zip = zips / f"{LEAD}.zip"
    C.verify_parent(lead_zip, LEAD_SHA)
    six = scratch / "six"
    with ZipFile(lead_zip) as z:
        z.extractall(six)
    blob = (folder / "team_b.pt").read_bytes()
    source = patch_bank((six / "submission.py").read_text(), "ba_six", blob)
    (six / "submission.py").write_text(source)
    (six / "team_b.pt").write_bytes(blob)
    control = load_folder(six, data, "cpu")
    team_a_zero = G.zero(control, module, folder, data, "cpu", False)
    hard_mod, hard_dir = G.unpack(zips, f"{BASE}.zip", BASE_SHA, scratch)
    hard = C.load(hard_mod, hard_dir, data, "cpu")
    with ExitStack() as hooks:
        hooks.enter_context(patch.object(module.TerminalTeamAStack, "TEAM_A_MIX", 0.0))
        hooks.enter_context(patch.object(module.TeamBlendStack, "TEAM_B_MIX", 0.0))
        disabled = C.load(module, folder, data, "cpu")
    assert disabled.team_a is None and disabled.team_b is None
    hard_zero = {}
    for session, rows in C.D.plans(data, 71120, C.ZERO_STREAMS).items():
        x = torch.from_numpy(data["X"][rows])
        a, b = G.play(hard, x, capture=True), G.play(disabled, x, capture=True)
        for av, bv in zip(a[:2], b[:2], strict=True):
            if av.dtype != bv.dtype or av.shape != bv.shape or av.tobytes() != bv.tobytes():
                raise ValueError(f"{session}: full zero differs from signed hard")
        if a[2] != b[2]:
            raise ValueError(f"{session}: full zero changed stage activity")
        hard_zero[session] = len(rows)
    assert sum(hard_zero.values()) == sum(team_a_zero.values()) == 379
    return dict(team_a_zero=team_a_zero, full_zero=hard_zero)


def active(module, folder, data, smoke):
    models = [C.load(module, folder, data, "cuda:0") for _ in range(2)]
    for m in models:
        G.clean(m)
        fused = [net.body for net in m.team_b.nets if getattr(net.body, "fused", None) is not None]
        if (len(fused) != 4 or len(m.team_b.nets) != 6 or m.TEAM_A_MIX != 0.10 or m.TEAM_B_MIX != 0.25):
            raise ValueError("fused bank, MANAS dose, or Team B dose differs")
        if m is models[1]:
            for body in fused:
                body.fused = None
    reports = {}
    for session, rows in C.D.plans(data, 71120, C.ZERO_STREAMS).items():
        x = torch.from_numpy(data["X"][rows[:72] if smoke else rows]).to("cuda:0")
        a, b = [G.play(m, x, capture=True) for m in models]
        np.testing.assert_array_equal(a[0], b[0])
        if a[2] != b[2]:
            raise ValueError(f"{session}: fused/unfused changed terminal work")
        if smoke:
            prefix, _, _ = G.play(models[0], x[:32], capture=True)
            np.testing.assert_array_equal(prefix, a[0][:32])
        reports[session] = dict(windows=len(x), changed=0,
                                max_abs=float(np.max(np.abs(a[1] - b[1]))), activity=a[2])
    assert smoke or sum(r["windows"] for r in reports.values()) == 379
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("zero", "smoke", "identity"))
    p.add_argument("candidate", type=Path)
    p.add_argument("sha")
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    C.verify_parent(a.data, C.DATA_SHA)
    data = C.D.load(a.data)
    with TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        zips = a.candidate.parent
        mod, folder = G.unpack(zips, a.candidate.name, a.sha, scratch)
        results = (zero(zips, mod, folder, data, scratch) if a.stage == "zero" else
                   active(mod, folder, data, a.stage == "smoke"))
    a.out.write_text(json.dumps(dict(sha=a.sha, stage=a.stage, results=results), indent=1) + "\n")
    print(f"PASS {a.stage}", flush=True)


if __name__ == "__main__":
    main()
