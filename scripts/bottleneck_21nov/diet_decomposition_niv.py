"""Run the four exact source arms in the existing Scherer NI-V worker."""
import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from diet_decomposition_patch import ARMS, patch
from quarter_niv import FIRST, PARENT_SHA
from returning_em_niv import CANDIDATE, PIN, validate_pair
from build_cb_speed import digest


def score(mode, task, arm, enrolled_sha, baseline, pinned, root):
    sys.path.insert(0, str(Path.cwd() / "niv"))
    import worker

    returning = mode != "smoke-new"
    smoke = mode.startswith("smoke")
    worker.orders.SEEDS = tuple(range(PIN if smoke else FIRST, (PIN if smoke else FIRST) + 8))
    os.environ["NIV_VISITS"] = "2" if returning else "1,2"
    if digest(CANDIDATE) != PARENT_SHA:
        raise ValueError("wrong Scherer-free em_half parent")
    overlay = Path("/enrolled.zip")
    if returning and digest(overlay) != enrolled_sha:
        raise ValueError("wrong Scherer enrolment overlay")
    name = f"smoke_{task}.npz" if smoke else f"p{task}_c{task % 4}.npz"
    tag = f"ret_trial_k{task}" if returning else "trial"
    reference = pinned / "active" / tag / name if smoke else baseline / f"ret_lead_k{task}" / name
    original_load = worker.load
    variants = ("half", *ARMS) if smoke else (arm,)
    for variant in variants:
        def load(directory, channels):
            if returning:
                with ZipFile(overlay) as enrolled:
                    (directory / "weights.npz").write_bytes(enrolled.read("weights.npz"))
            if variant != "half":
                source = directory / "submission.py"
                source.write_text(patch(source.read_text(), ARMS[variant]))
            return original_load(directory, channels)

        worker.load = load
        folder = root / variant / tag
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / name
        if output.exists():
            raise FileExistsError(output)
        orders = "full:0" if smoke else ",".join(f"{s}:{i}" for s in ("full", "within") for i in range(8))
        with TemporaryDirectory(dir=folder) as tmp:
            temporary = Path(tmp) / name
            worker.main(CANDIDATE, temporary, str(task), orders)
            counts = validate_pair(reference, temporary)
            for scheme, audit in counts.items():
                expected = 1 if smoke and returning else (2 if smoke else 8)
                if audit["streams"] != expected or audit["recognised"] != (expected if returning else 0):
                    raise ValueError(f"{variant}: wrong coverage for {scheme}: {audit}")
                if (variant == "half" or not returning) and audit["changed"]:
                    raise ValueError(f"{variant}: pinned/zero mismatch for {scheme}: {audit}")
            temporary.rename(output)
        print(f"PASS {variant} {mode} person {task}: {counts}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke-ret", "smoke-new", "score"))
    parser.add_argument("task", type=int)
    parser.add_argument("arm")
    parser.add_argument("enrolled_sha")
    for name in ("baseline", "pinned", "root"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    score(args.mode, args.task, args.arm, args.enrolled_sha, args.baseline, args.pinned, args.root)


if __name__ == "__main__":
    main()
