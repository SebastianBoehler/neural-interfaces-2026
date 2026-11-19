"""Authenticate each §257 warm control before reporting full within0 parity."""

import argparse
import json
import math

import numpy as np

from refit_smoke import HERE, ROOT, TOL, file_sha
from restore_refit import RELEASE
from within0_warm import ACTORS, CHAIN, FROZEN_SHA, FTREL, ORDER, PEOPLE, RELAB, rows_sha


V1 = HERE.parent / "within0_warm_v1"
V1_SHA = "ca5607d8fd894c5dca2d3f9271d3a1f53404b9dcc794d95653adaded99c908aa"


def references():
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as data:
        subject, visit = data["subject"], data["session"]
    streams = np.char.add(np.char.add(subject, "|"), visit)
    return subject, streams, file_sha(HERE / "source.sha256")


def validate(path, family, refs, smoke, smoke_sha=None, deterministic=False, repeat_sha=None, recover=()):
    subject, streams, source = refs
    if family != "fbl":
        source = V1_SHA
    if file_sha(path.parents[1] / "source.sha256") != source:
        raise ValueError("warm release manifest differs")
    d = json.loads(path.read_text())
    if (d["source"] != source or d["family"] != family or d["order"] != ORDER or d["held_labels_opened"]
            or d["smoke"] != smoke or d["smoke_sha256"] != smoke_sha
            or (not d["archive_parity"] and not deterministic and not recover)
            or d["frozen_sha256"] != FROZEN_SHA or path.stat().st_mode & 0o222
            or d["chain_sha256"] != file_sha(CHAIN) or d["relab_sha256"] != file_sha(RELAB)
            or d["actor_sha256"] != file_sha(FTREL) or "RTX 2080 Ti" not in d["gpu"]):
        raise ValueError(f"unqualified authentic warm receipt {path}")
    if family == "fbl" and (d["deterministic"] != deterministic or d["repeat_sha256"] != repeat_sha):
        raise ValueError("FBLight runtime proof mismatch")
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
    failed_folds = set()
    for visit, r in d["sessions"].items():
        n = int(np.count_nonzero(streams == visit))
        if (r["rows"] != n or r["steps"] != 10 * len(range(20, n, 10)) or r["prefix_steps"] != 60
                or not all(r[key] for key in ("first_20_exact", "strict_targets_exact"))
                or not 1e-3 < r["warm_movement_abs"] <= 1 or not 1e-8 < r["first_active_abs"] <= 1):
            raise ValueError(f"wrong activity or strict prefix proof for {visit}")
        if (family != "fbl" or deterministic) and not r["interleaved_reset_exact"]:
            raise ValueError(f"bitwise reset proof missing for {visit}")
        if family == "fbl":
            bound = 0. if deterministic else TOL
            for key in ("consecutive_repeat", "interleaved_repeat"):
                v = r[key]
                if v["changed"] or not math.isfinite(v["max_abs"]) or not 0 <= v["max_abs"] <= bound:
                    raise ValueError(f"FBLight reset/repeat parity failed for {visit}")
        for key in ("base", "warm", "prefix"):
            v = r[key]
            if deterministic and key != "prefix":
                continue
            if v["changed"] or not math.isfinite(v["max_abs"]) or not 0 <= v["max_abs"] <= TOL:
                person = visit.split("|")[0].split("/")[-1]
                if key == "warm" and person in recover and math.isfinite(v["max_abs"]) and 0 <= v["max_abs"] <= 1:
                    failed_folds.add(person)
                    continue
                raise ValueError(f"historical {key} parity failed for {visit}")
    if failed_folds != set(recover):
        raise ValueError("the original warm failure scope differs from the registered source recovery")
    return file_sha(path), d["sessions"]


def collect():
    refs = references()
    outputs = {}
    for family in ACTORS.values():
        release = HERE if family == "fbl" else V1
        deterministic = None
        if family == "fbl":
            deterministic, _ = validate(release / "deterministic/fbl.json", family, refs, smoke=True, deterministic=True)
        one, first = validate(release / "smoke" / f"{family}.json", family, refs, smoke=True)
        all_sha, rest = validate(release / "members" / f"{family}.json", family, refs, smoke=False,
                                 smoke_sha=one, repeat_sha=deterministic)
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
        if family == "fbl":
            outputs[family].update(deterministic_reset_receipt=deterministic,
                                   max_consecutive_abs=max(r["consecutive_repeat"]["max_abs"] for r in visits.values()),
                                   max_interleaved_abs=max(r["interleaved_repeat"]["max_abs"] for r in visits.values()))
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
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.smoke_task is None:
        collect()
    else:
        name = ACTORS[args.smoke_task]
        folder = "deterministic" if args.deterministic else "smoke"
        print(validate(HERE / folder / f"{name}.json", name, references(),
                       smoke=True, deterministic=args.deterministic)[0], flush=True)
