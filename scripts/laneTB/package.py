"""Build the selected new ZIP and check its zero-weight path on CPU."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from run import D, stream_session, timed_load
from build import archive, sha


def zero_check(base, candidate, data, out):
    d = D.load(data)
    signals = torch.from_numpy(d["X"])
    with TemporaryDirectory(dir=out.parent) as tmp:
        tmp = Path(tmp)
        a, b = tmp / "base", tmp / "candidate"
        with ZipFile(base) as z:
            z.extractall(a)
        with ZipFile(candidate) as z:
            z.extractall(b)
        sub = b / "submission.py"
        text = sub.read_text()
        weight_line = next(s for s in text.splitlines() if s.startswith("    TEAM_B_MIX = "))
        assert weight_line != "    TEAM_B_MIX = 0.0" and text.count(weight_line) == 1
        sub.write_text(text.replace(weight_line, "    TEAM_B_MIX = 0.0"))
        ref, _ = timed_load(a, d, "cpu")
        zero, _ = timed_load(b, d, "cpu")
        assert zero.team_b is None
        counts = {}
        for s, rows in D.plans(d, 71120, ["P01|1", "P04|2"]).items():
            old, old_taps, _ = stream_session(ref, signals, rows, "cpu", capture=True)
            new, new_taps, _ = stream_session(zero, signals, rows, "cpu", capture=True)
            np.testing.assert_array_equal(old, new)
            np.testing.assert_array_equal(old_taps, new_taps)
            counts[s] = len(rows)
            print("CPU blend zero", s, len(rows), "windows; identical mixer, step output and labels", flush=True)
    out.write_text(json.dumps(dict(archive_sha256=sha(candidate.read_bytes()), cpu_zero=counts), indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    arm = json.loads(a.result.read_text())["package_arm"]
    candidate = a.outdir / f"final_stack_hardcb_safe_team_b_{arm}.zip"
    archive(a.base, candidate, arm, a.bank)
    zero_check(a.base, candidate, a.data, a.outdir / "package.json")
