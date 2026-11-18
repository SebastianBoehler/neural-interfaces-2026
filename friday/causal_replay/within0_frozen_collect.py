"""Seal the complete ten-fold within0 frozen eligibility diagnostic."""

import argparse
import json
import math

import numpy as np

from member_inputs import ORDERS, pinned_delivery
from refit_smoke import HERE, ROOT, TOL, file_sha
from restore_refit import RELEASE
from within0_frozen_probe import ARCHS, FRESH, FRESH_SOURCE, ORDER, PEOPLE, rows_sha


P01 = HERE.parent / "within0_frozen_probe_v2/p01.json"
P01_SOURCE = "264cb49adf604532daa93d1a2e25d4a9c15f78df4e9327c14780752636021ebb"
P04 = HERE.parent / "within0_frozen_probe_v3/p04.json"
P04_SOURCE = "d26ff02e909e30502cacf33ce0d2288d7734214a77f0e34e712d86ca2a01d206"


def finite_delta(row, n):
    if (type(row["changed"]) is not int or not 0 <= row["changed"] <= n
            or not math.isfinite(row["max_abs"]) or not 0 <= row["max_abs"] <= 1
            or not math.isfinite(row["prefix_max_abs"]) or not 0 <= row["prefix_max_abs"] <= TOL):
        raise ValueError("malformed posterior, decisions or temporal proof")


def references():
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as data:
        subject, visit = data["subject"], data["session"]
    if len(subject) != 2655 or sorted(set(subject)) != [f"Dreyer2026Proteus/{p}" for p in PEOPLE]:
        raise ValueError("held cohort differs from the original folds")
    delivery = pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORDERS.index(ORDER)]
    delivery_sha = rows_sha(delivery)
    original = json.loads((FRESH_SOURCE / "frozen.json").read_text())
    replay = json.loads((FRESH / "frozen_orders.json").read_text())
    source_fits = {family: json.loads((RELEASE / "members" / f"{family}_s0.json").read_text())
                   for family in ARCHS}
    return subject, visit, delivery_sha, original, replay, source_fits


def checked_fold(person, path, source, refs):
    subject, visit, delivery_sha, original, replay, source_fits = refs
    raw = json.loads(path.read_text())
    held = subject == f"Dreyer2026Proteus/{person}"
    n = int(held.sum())
    k = len(set(visit[held]))
    fit_sha = rows_sha(np.flatnonzero(~held))
    if ((raw["person"], raw["rows"], raw["order"], raw["prediction_mode"]) != (person, n, ORDER, "frozen")
            or raw["held_labels_opened"] or raw["delivery_sha256"] != delivery_sha
            or raw["source_sha256"] != source or path.stat().st_mode & 0o222
            or set(raw["candidates"]) != ARCHS.keys() or "RTX 2080 Ti" not in raw["gpu"]
            or raw["torch"] != "2.14.1+cu126"):
        raise ValueError(f"unexpected frozen receipt in {person}")
    if person != "P01" and (raw["held_rows_sha256"] != rows_sha(np.flatnonzero(held))
                            or raw["fit_rows_sha256"] != fit_sha or raw["session_count"] != k):
        raise ValueError(f"held or fit exclusions differ for {person}")
    for family, row in raw["candidates"].items():
        fit = source_fits[family]["fit_states"][person]
        finite_delta(row, n)
        passed = row["changed"] == 0 and row["max_abs"] <= TOL
        if (row["eligible"] != passed or not row["interleaved_reset_exact"]
                or row["fit_rows_sha256"] != fit_sha or row["source_checkpoint"] != fit["sha256"]):
            raise ValueError(f"unproved fit qualification {person}/{family}")
    row = raw["fresh_sinc"]
    finite_delta(row, n)
    if (row["source_bank"] != replay["bank_sha256"]
            or row["source_checkpoint"] != original["folds"][person]["checkpoint_sha256"]
            or row["active"] != (row["max_abs"] > 1e-8)):
        raise ValueError(f"fresh Sinc input differs for {person}")
    return raw, k


def check_smoke(refs):
    if file_sha(P04.parent / "source.sha256") != P04_SOURCE:
        raise ValueError("P04 smoke release differs")
    raw, sessions = checked_fold("P04", P04, P04_SOURCE, refs)
    if raw["rows"] != 566 or sessions != 3:
        raise ValueError("P04 multi-visit smoke scope differs")
    return raw, file_sha(P04)


def collect():
    refs = references()
    subject, _, delivery_sha, _, replay, _ = refs
    source_sha = file_sha(HERE / "source.sha256")
    smoke, smoke_sha = check_smoke(refs)
    checks, cases, sinc, sessions = {}, {family: {} for family in ARCHS}, {}, 0
    for person in PEOPLE:
        if person == "P04":
            raw, k, digest = smoke, smoke["session_count"], smoke_sha
        else:
            first = person == "P01"
            path = P01 if first else HERE / f"{person.lower()}.json"
            raw, k = checked_fold(person, path, P01_SOURCE if first else source_sha, refs)
            if not first and raw["p04_smoke_sha256"] != smoke_sha:
                raise ValueError(f"the P04 smoke did not qualify {person}")
            digest = file_sha(path)
        sessions += k
        for family, row in raw["candidates"].items():
            cases[family][person] = row
        sinc[person] = raw["fresh_sinc"]
        checks[person] = digest
    if sessions != 14:
        raise ValueError("unexpected number of held sessions")
    family_status = {}
    for family, slots in cases.items():
        failed = [p for p, r in slots.items() if not r["eligible"]]
        family_status[family] = dict(
            eligible=not failed, failed=failed, qualified_folds=len(PEOPLE) - len(failed),
            max_abs=max(r["max_abs"] for r in slots.values()), changed=sum(r["changed"] for r in slots.values()),
            max_prefix_abs=max(r["prefix_max_abs"] for r in slots.values()), interleaved_reset_exact=True,
            folds={p: {k: r[k] for k in ("eligible", "max_abs", "changed", "prefix_max_abs")}
                   for p, r in slots.items()},
        )
    sinc_max = max(r["max_abs"] for r in sinc.values())
    result = dict(order=ORDER, rows=len(subject), sessions=sessions, source=source_sha,
                  delivery_sha256=delivery_sha, receipts=checks, families=family_status, p04_smoke_sha256=smoke_sha,
                  fresh_sinc=dict(source_bank=replay["bank_sha256"], same_state_layout_exact=True,
                                  active=sinc_max > 1e-8, max_abs=sinc_max, changed=sum(r["changed"] for r in sinc.values()),
                                  max_prefix_abs=max(r["prefix_max_abs"] for r in sinc.values())),
                  frozen_inputs_eligible=all(v["eligible"] for v in family_status.values()) and sinc_max > 1e-8,
                  historical_refit_parity_verified=False, held_labels_opened=False, terminal_score_allowed=False)
    target = HERE / "qualification.json"
    with target.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    target.chmod(0o444)
    print("PASS complete frozen diagnostic", file_sha(target), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("p04", "collect"))
    if parser.parse_args().mode == "p04":
        print(check_smoke(references())[1], flush=True)
    else:
        collect()
