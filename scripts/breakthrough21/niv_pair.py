"""Run an exact Scherer-free candidate and fused lead on the same NI-V rows."""
import argparse
from hashlib import file_digest
import json
from os import environ
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

LEAD_SHA = "7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07"


def check_file(path, expected):
    with Path(path).open("rb") as f:
        if file_digest(f, "sha256").hexdigest() != expected:
            raise ValueError(f"unexpected NI-V artifact {path}")


def paired(mode, task, base, trial, sha, root, first, receipt):
    import worker

    worker.orders.SEEDS = tuple(range(first, first + 8))
    returning = mode in ("ret", "smoke-ret")
    person = task if returning else task // 4
    environ["NIV_VISITS"] = "2" if returning else "1,2"
    if mode.startswith("smoke"):
        indices = [0]
        schemes = ("full",)
    else:
        indices = range(8) if returning else (2 * (task % 4), 2 * (task % 4) + 1)
        schemes = ("full", "within")
    arms = ",".join(f"{scheme}:{k}" for scheme in schemes for k in indices)
    check_file(base, LEAD_SHA)
    check_file(trial, sha)
    if returning:
        enrolled = Path("/enrolled.zip")
        check_file(enrolled, receipt)
        original = worker.load

        def load(directory, channels):
            # Keep every difference confined to the terminal arm or population prior.
            with ZipFile(enrolled) as z:
                (directory / "weights.npz").write_bytes(z.read("weights.npz"))
            return original(directory, channels)

        worker.load = load
    outputs = []
    for label, archive in (("lead", base), ("trial", trial)):
        tag = f"ret_{label}_k{person}" if returning else label
        folder = root / tag
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (f"smoke_{person}.npz" if mode.startswith("smoke") else f"p{person}_c{task % 4}.npz")
        if path.exists():
            raise FileExistsError(path)
        worker.main(archive, path, str(person), arms)
        outputs.append(path)
    with np.load(outputs[0], allow_pickle=False) as a, np.load(outputs[1], allow_pickle=False) as b:
        if a.files != b.files:
            raise ValueError("unmatched NI-V paths")
        for key in a.files:
            if key.endswith("|rows"):
                np.testing.assert_array_equal(a[key], b[key])


def report(root, first):
    import collect as C

    C.bank.orders.SEEDS = tuple(range(first, first + 8))
    data = "/external.npz"
    for role in ("trial", "lead"):
        for rows, expected in ((C.cohort_b_new(root, role, data), 288),
                               (C.cohort_b_ret(root, role, data), 144)):
            if rows is None or np.isfinite(rows["cell"]).sum() != expected:
                raise ValueError(f"NI-V coverage differs for {role}")
    out = root / "report.json"
    if out.exists():
        raise FileExistsError(out)
    sys.argv = ["collect.py", str(root), str(out), "--a", "trial", "--b", "lead", "--data", data]
    C.main()
    result = json.loads(out.read_text())
    result["headline"] = f"paired full-session shuffle, seeds {first}–{first + 7}"
    out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("new", "ret", "smoke-new", "smoke-ret", "report"))
    p.add_argument("root", type=Path)
    p.add_argument("first", type=int)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--base", type=Path)
    p.add_argument("--trial", type=Path)
    p.add_argument("--sha", default="")
    p.add_argument("--enrolled-sha", default="")
    a = p.parse_args()
    sys.path.insert(0, str(Path.cwd() / "niv"))
    if a.mode == "report":
        report(a.root, a.first)
    else:
        paired(a.mode, a.task, a.base, a.trial, a.sha, a.root, a.first, a.enrolled_sha)


if __name__ == "__main__":
    main()
