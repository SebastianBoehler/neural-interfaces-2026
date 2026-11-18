"""Authenticate each §257 warm control before reporting full within0 parity."""

import argparse
import json
import math

import numpy as np

from refit_smoke import HERE, ROOT, TOL, file_sha
from restore_refit import RELEASE
from within0_warm import ACTORS, CHAIN, FROZEN_SHA, FTREL, ORDER, PEOPLE, RELAB, rows_sha


def references():
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as data:
        subject, visit = data["subject"], data["session"]
    streams = np.char.add(np.char.add(subject, "|"), visit)
    return subject, streams, file_sha(HERE / "source.sha256")


def validate(path, family, refs, smoke, smoke_sha=None):
    subject, streams, source = refs
    d = json.loads(path.read_text())
    if (d["source"] != source or d["family"] != family or d["order"] != ORDER or d["held_labels_opened"]
            or d["smoke"] != smoke or d["smoke_sha256"] != smoke_sha or not d["archive_parity"]
            or d["frozen_sha256"] != FROZEN_SHA or path.stat().st_mode & 0o222
            or d["chain_sha256"] != file_sha(CHAIN) or d["relab_sha256"] != file_sha(RELAB)
            or d["actor_sha256"] != file_sha(FTREL) or "RTX 2080 Ti" not in d["gpu"]):
        raise ValueError(f"unqualified authentic warm receipt {path}")
    people = PEOPLE[:1] if smoke else PEOPLE[1:]
    fit = json.loads((RELEASE / "members" / f"{family}_s0.json").read_text())["fit_states"]
    keep = np.isin(subject, [f"Dreyer2026Proteus/{p}" for p in people])
    if set(d["sessions"]) != set(streams[keep]) or set(d["fits"]) != set(people):
        raise ValueError("warm receipt scopes the wrong held folds")
    for p in people:
        mask = subject != f"Dreyer2026Proteus/{p}"
        r = d["fits"][p]
        if r != fit[p] or r["fit_rows_sha"] != rows_sha(np.flatnonzero(mask)):
            raise ValueError(f"unproved source exclusion for {p}")
    for visit, r in d["sessions"].items():
        n = int(np.count_nonzero(streams == visit))
        if (r["rows"] != n or r["steps"] != 10 * len(range(20, n, 10)) or r["prefix_steps"] != 60
                or not all(r[key] for key in ("first_20_exact", "interleaved_reset_exact", "strict_targets_exact"))
                or not 1e-3 < r["warm_movement_abs"] <= 1 or not 1e-8 < r["first_active_abs"] <= 1):
            raise ValueError(f"wrong activity or strict prefix proof for {visit}")
        for key in ("base", "warm", "prefix"):
            v = r[key]
            if v["changed"] or not math.isfinite(v["max_abs"]) or not 0 <= v["max_abs"] <= TOL:
                raise ValueError(f"historical {key} parity failed for {visit}")
    return file_sha(path), d["sessions"]


def collect():
    refs = references()
    outputs = {}
    for family in ACTORS.values():
        one, first = validate(HERE / "smoke" / f"{family}.json", family, refs, smoke=True)
        all_sha, rest = validate(HERE / "members" / f"{family}.json", family, refs, smoke=False, smoke_sha=one)
        visits = first | rest
        count = sum(r["rows"] for r in visits.values())
        steps = sum(r["steps"] for r in visits.values())
        if (count, len(visits), steps) != (2655, 14, 2400):
            raise ValueError(f"incomplete historical warm proof for {family}")
        outputs[family] = dict(archive_parity=True, warm_max_abs=max(r["warm"]["max_abs"] for r in visits.values()),
                               warm_changed=sum(r["warm"]["changed"] for r in visits.values()), steps=steps,
                               base_max_abs=max(r["base"]["max_abs"] for r in visits.values()),
                               prefix_max_abs=max(r["prefix"]["max_abs"] for r in visits.values()),
                               minimum_movement=min(r["warm_movement_abs"] for r in visits.values()),
                               p01_receipt=one, remaining_receipt=all_sha)
    summary = dict(held_labels_opened=False, source=refs[2], order=ORDER, rows=2655, visits=14,
                   families=outputs, frozen_sha256=FROZEN_SHA, chain_sha256=file_sha(CHAIN),
                   relab_sha256=file_sha(RELAB), legacy_sinc_qualified=False, terminal_score_allowed=False)
    path = HERE / "qualification.json"
    with path.open("x") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    print("PASS historical warm", file_sha(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_task", nargs="?", type=int, choices=ACTORS)
    args = parser.parse_args()
    if args.smoke_task is None:
        collect()
    else:
        name = ACTORS[args.smoke_task]
        print(validate(HERE / "smoke" / f"{name}.json", name, references(), smoke=True)[0], flush=True)
