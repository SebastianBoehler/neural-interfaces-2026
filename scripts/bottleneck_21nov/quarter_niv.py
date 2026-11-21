"""Compare the source-only quarter diet on the pinned Scherer-free NI-V setup."""
from functools import partial
import json
from pathlib import Path
import sys

from returning_diet import ARMS, patch
from returning_em import build
from returning_em_niv import CANDIDATE, PIN, validate_pair
from build_cb_speed import digest

FIRST = 71864
PARENT = CANDIDATE
TRIAL = PARENT.with_stem(PARENT.stem + "_diet_quarter")
PARENT_SHA = "376e0f26ac1660e8330df8a0100a74cc82b3f3b0858a7a77433de7149bac823b"
RESULTS = Path("niv-results")


def score(mode, task, enrolled_sha, pinned, candidate_sha):
    import niv_pair

    niv_pair.LEAD_SHA = PARENT_SHA
    if mode.startswith("smoke"):
        folder = RESULTS / "smoke"
        niv_pair.paired(mode, task, PARENT, TRIAL, candidate_sha, folder, PIN, enrolled_sha)
        returning = mode == "smoke-ret"
        lead, trial = ("ret_lead_k5", "ret_trial_k5") if returning else ("lead", "trial")
        name = "smoke_5.npz"
        reference = pinned / "active" / trial / name
        if any(item["changed"] for item in validate_pair(reference, folder / lead / name).values()):
            raise ValueError("em_half no longer matches its pinned NI-V receipt")
        counts = validate_pair(folder / lead / name, folder / trial / name)["full"]
        if returning and (not counts["recognised"] or not counts["changed"]):
            raise ValueError(f"returning smoke did not exercise the quarter diet: {counts}")
        if not returning and (counts["recognised"] or counts["changed"]):
            raise ValueError(f"new-person smoke failed the identity control: {counts}")
        print(f"OK pinned em_half and quarter checks: {mode} {counts}", flush=True)
    else:
        niv_pair.paired(mode, task, PARENT, TRIAL, candidate_sha, RESULTS, FIRST, enrolled_sha)


def report(mode, candidate_sha):
    import niv_pair
    import returning_em_niv

    if mode == "report":
        niv_pair.report(RESULTS, FIRST)
        return
    cohort = mode.removeprefix("report-")
    result = returning_em_niv.report(cohort)
    result.update(first=FIRST, last=FIRST + 7, parent_sha=PARENT_SHA, candidate_sha=candidate_sha)
    Path(f"out/{mode}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "niv")]
    mode, task, enrolled_sha, pinned = sys.argv[1:]
    receipt = Path("candidate.sha")
    if mode == "build":
        build(PARENT, TRIAL, expected_sha=PARENT_SHA, patch_source=partial(patch, weight=ARMS["quarter"]))
        receipt.write_text(digest(TRIAL) + "\n")
        return
    candidate_sha = receipt.read_text().strip()
    if mode.startswith("report"):
        report(mode, candidate_sha)
    else:
        score(mode, int(task), enrolled_sha, Path(pinned), candidate_sha)


if __name__ == "__main__":
    main()
