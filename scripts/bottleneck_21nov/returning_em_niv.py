"""Qualify the recognition-gated SessionEM patch on the Scherer-free FP16 package."""
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

from returning_em import build, patch
from build_cb_speed import digest

FIRST = 71856
PIN = 71840
PARENT_SHA = "0804a2f272d4fb4b6215cfcd7f282f0c813cbe4b4888f3f6b5f38f74964d6160"
PARENT = Path("zips/ba_six_fc_niv_fp16_turing.zip")
CANDIDATE = PARENT.with_stem(PARENT.stem + "_recognised_em_half")
RESULTS = Path("niv-results")


def validate_pair(parent_path, candidate_path):
    audit = {}
    with np.load(parent_path, allow_pickle=False) as parent, np.load(candidate_path, allow_pickle=False) as candidate:
        assert parent.files == candidate.files
        for key in parent.files:
            if not key.endswith("|rows"):
                continue
            stream = key.removesuffix("|rows")
            np.testing.assert_array_equal(parent[key], candidate[key])
            rec, routed, err = map(int, parent[stream + "|meta"][:3])
            np.testing.assert_array_equal([rec, routed, err], candidate[stream + "|meta"][:3])
            a, b = parent[stream + "|labels"], candidate[stream + "|labels"]
            if rec >= len(a):
                raise ValueError(f"invalid recognition index {rec} on {stream}")
            guard = len(a) if rec == -1 else rec
            np.testing.assert_array_equal(a[:guard], b[:guard])
            scheme = stream.split("|", 1)[0]
            counts = audit.setdefault(scheme, dict(streams=0, recognised=0, windows=0, changed=0))
            counts["streams"] += 1
            counts["recognised"] += rec >= 0
            counts["windows"] += len(a)
            counts["changed"] += int(np.count_nonzero(a != b))
    return audit


def score(mode, task, enrolled_sha, pinned):
    import niv_pair
    import worker

    niv_pair.LEAD_SHA = PARENT_SHA
    candidate_sha = Path("candidate.sha").read_text().strip()

    def run(root, seed):
        niv_pair.paired(mode, task, PARENT, CANDIDATE, candidate_sha, root, seed, enrolled_sha)

    if mode.startswith("smoke"):
        count = 0
        original_load = worker.load
        zero = True

        def load(directory, channels):
            model = original_load(directory, channels)
            step = type(model.routeB.tail.em).step
            emit = step.__globals__.get("returning_em_output")
            if emit is not None:
                def instrument(p, q, weight, active):
                    nonlocal count
                    count += weight == 0.5 and active
                    return emit(p, q, 1.0 if zero else weight, active)

                step.__globals__["returning_em_output"] = instrument
            return model

        worker.load = load
        run(RESULTS / "zero", PIN)
        returning = mode == "smoke-ret"
        lead, trial, reference = ("ret_lead_k5", "ret_trial_k5", "ret_trial_k5") if returning else ("lead", "trial", "trial")
        relative = Path("smoke_5.npz")
        control = RESULTS / "zero" / lead / relative
        for path in (pinned / reference / relative, RESULTS / "zero" / trial / relative):
            audit = validate_pair(control, path)
            if any(item["changed"] for item in audit.values()):
                raise ValueError(f"pinned/zero labels changed: {path}")
        zero = False
        count = 0
        worker.load = load  # the pinned returning run installed its own overlay wrapper
        run(RESULTS / "active", PIN)
        audit = validate_pair(RESULTS / "active" / lead / relative, RESULTS / "active" / trial / relative)
        if returning and not count:
            raise ValueError("pinned returning stream did not exercise active half weight")
        print(f"OK {mode} pinned and disabled identity, active emissions={count}, paired={audit}", flush=True)
    else:
        run(RESULTS, FIRST)


def report(mode):
    import collect as c

    returning = mode == "ret"
    audit = {}
    for task in range(9 if returning else 36):
        name = f"p{task}_c{task % 4}.npz" if returning else f"p{task // 4}_c{task % 4}.npz"
        lead, trial = (f"ret_lead_k{task}", f"ret_trial_k{task}") if returning else ("lead", "trial")
        for scheme, counts in validate_pair(RESULTS / lead / name, RESULTS / trial / name).items():
            total = audit.setdefault(scheme, dict.fromkeys(counts, 0))
            for key, value in counts.items():
                total[key] += value
    expected = 72 if returning else 144
    if set(audit) != {"full", "within"} or any(item["streams"] != expected for item in audit.values()):
        raise ValueError(f"incomplete {mode} audit: {audit}")
    cohort = c.cohort_b_ret if returning else c.cohort_b_new
    a, b = (cohort(RESULTS, role, "/external.npz")["cell"] for role in ("trial", "lead"))
    if any(np.isfinite(cell).sum() != expected * 2 for cell in (a, b)):
        raise ValueError(f"incomplete {mode} scoring coverage")
    return dict(audit=audit, trial=c.absolute(a), parent=c.absolute(b), delta=c.stats.report(a, b))


if __name__ == "__main__":
    sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "niv")]
    mode = sys.argv[1]
    if mode == "build":
        build(PARENT, CANDIDATE, expected_sha=PARENT_SHA, patch_source=partial(patch, weight=0.5))
        Path("candidate.sha").write_text(digest(CANDIDATE) + "\n")
    elif mode.startswith("report-"):
        cohort = mode.removeprefix("report-")
        output = dict(first=FIRST, last=FIRST + 7, parent_sha=PARENT_SHA,
                      candidate_sha=Path("candidate.sha").read_text().strip(), **report(cohort))
        Path(f"out/{mode}.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    else:
        score(mode, int(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
