"""Score the frozen gate on the fixed held-out participants."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "brainhero_wave"))
from score_brainhero import metrics

from actual_gate import decisions
from plan import EVAL, LATE, metadata, partition, protocol


def slice_stats(m, person=None):
    people = m["people"]
    chosen = [people[person]] if person else list(people.values())
    return {key: float(np.mean([x[key] for x in chosen])) for key in ("delta", "score")}


def run(path, output):
    meta = metadata()
    _, target, later = partition(meta)
    with np.load(path / "outer.npz", allow_pickle=False) as outer:
        trace = dict(outer)
    np.testing.assert_array_equal(trace["indices"], target)
    gate = json.loads((path / "gate.json").read_text())
    predicted, broad, runner = decisions(trace, gate)
    # Labels first enter this separate scoring stage, after gate and outer inference are fixed.
    with np.load(protocol.DATA, allow_pickle=False) as dataset:
        meta["y"] = dataset["y"]
    truth = meta["y"][target]
    at_later = np.searchsorted(target, later)
    np.testing.assert_array_equal(target[at_later], later)
    cohorts = {"unseen": np.arange(len(target)), "later_unseen": at_later}
    winner = trace["final"].argmax(1)
    result = {}
    for arm, labels in predicted.items():
        detail = {}
        assert np.all((labels == winner) | (broad & (labels == runner)))
        for name, at in cohorts.items():
            m = metrics(meta, target[at], trace["final"][at], np.eye(3)[labels[at]])
            changed = labels[at] != winner[at]
            detail[name] = {
                "all": slice_stats(m), "P10": slice_stats(m, "P10"),
                "interfaces": {k: slice_stats({"people": v}) for k, v in m["people_by_interface"].items()},
                "participants": m["people"], "windows": len(at), "eligible": int(broad[at].sum()),
                "switches": int(changed.sum()),
                "rescues": int((changed & (labels[at] == truth[at])).sum()),
                "regressions": int((changed & (winner[at] == truth[at])).sum()),
                "active_routes": int((trace["routing"][at, 2] > 0).sum()),
            }
        result[arm] = {"J_delta": float(np.mean([v["all"]["delta"] for v in detail.values()])), **detail}
    report = {"eval_participants": EVAL, "later_unseen_participants": LATE,
              "scope": "fixed participant-disjoint public evaluation; later slice overlaps unseen",
              "source_fit_rows": {k: v["fit_rows"] for k, v in gate.items()},
              "source_outcomes": {k: v["outcomes"] for k, v in gate.items()},
              "arms": result}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("artifacts", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.artifacts, args.output)
