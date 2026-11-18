"""Seal the historical within0 warm bank only with authenticated FBLight recovery."""

import json

import numpy as np

from refit_smoke import HERE, TOL, file_sha
from within0_warm import ACTORS, CHAIN, FROZEN_SHA, FTREL, ORDER, RELAB, rows_sha
from within0_warm_collect import V1, references, validate


V2 = HERE.parent / "within0_warm_v2"
V2_SHA = "765b3609881377b42591936806d022f9f8c0bd439823882ba96d496c9a81c17f"
RECOVER = HERE.parent / "within0_warm_recovery_v1"
FAILED = "d720d1fbf91bb85bdacbe1974a16c30140437322597b797aa7d6bd0121703e99"
FOLDS = ("P08", "P10")


def recover_one(person, subject, streams, source):
    path = RECOVER / f"fbl_s0_{person}.json"
    d = json.loads(path.read_text())
    held = subject == f"Dreyer2026Proteus/{person}"
    fit_sha = rows_sha(np.flatnonzero(~held))
    expected = dict(order=ORDER, person=person, source=source, original_failed_sha256=FAILED,
                    actor_sha256=file_sha(FTREL), frozen_sha256=FROZEN_SHA, chain_sha256=file_sha(CHAIN),
                    relab_sha256=file_sha(RELAB), held_labels_opened=False, fit_rows_sha=fit_sha,
                    fit_rows=int((~held).sum()), qualified=True)
    checkpoint = path.with_suffix(".pt")
    if (any(d[k] != v for k, v in expected.items()) or d["node"] != "galvani-cn114"
            or "RTX 2080 Ti" not in d["gpu"] or file_sha(checkpoint) != d["checkpoint_sha256"]
            or set(d["visits"]) != set(streams[held]) or path.stat().st_mode & 0o222):
        raise ValueError(f"the recovered RTX state differs for {person}")
    for visit, rec in d["visits"].items():
        n = int((streams == visit).sum())
        if (rec["rows"] != n or rec["steps"] != 10 * len(range(20, n, 10)) or rec["prefix_steps"] != 60
                or not rec["first_20_exact"] or not rec["strict_targets_exact"] or rec["warm_movement_abs"] <= 1e-3
                or rec["first_active_abs"] <= 1e-8):
            raise ValueError("the recovered state is inactive or misses historical visits")
        if any(rec[k]["changed"] or not 0 <= rec[k]["max_abs"] <= TOL for k in
               ("base", "warm", "prefix", "consecutive_repeat", "interleaved_repeat")):
            raise ValueError("the recovered state does not qualify numerically")
    return d["visits"], dict(checkpoint=str(checkpoint), checkpoint_sha256=d["checkpoint_sha256"],
                             fit_rows_sha=fit_sha, receipt_sha256=file_sha(path))


def collect(source):
    if file_sha(RECOVER / "source.sha256") != source:
        raise ValueError("recovery source manifest differs")
    subject, streams, collector_sha = references()
    refs = subject, streams, V2_SHA
    summaries, restored = {}, {}
    for family in ACTORS.values():
        release = V2 if family == "fbl" else V1
        recovered, deterministic = (), None
        if family == "fbl":
            recovered = FOLDS
            deterministic, _ = validate(release / "deterministic/fbl.json", family, refs, smoke=True, deterministic=True)
        one, first = validate(release / "smoke" / f"{family}.json", family, refs, smoke=True)
        all_sha, rest = validate(release / "members" / f"{family}.json", family, refs,
                                 smoke=False, smoke_sha=one, repeat_sha=deterministic, recover=recovered)
        if family == "fbl":
            if all_sha != FAILED:
                raise ValueError("FBLight prior failure receipt differs")
            for p in FOLDS:
                updated, restored[p] = recover_one(p, subject, streams, source)
                rest.update(updated)
        rows = first | rest
        steps = sum(r["steps"] for r in rows.values())
        if (len(rows), sum(r["rows"] for r in rows.values()), steps) != (14, 2655, 2400):
            raise ValueError("complete package of historical warm controls is missing")
        summaries[family] = dict(archive_parity=True, warm_max_abs=max(r["warm"]["max_abs"] for r in rows.values()),
                                 warm_changed=sum(r["warm"]["changed"] for r in rows.values()), steps=steps,
                                 prefix_max_abs=max(r["prefix"]["max_abs"] for r in rows.values()),
                                 p01_receipt=one, remaining_receipt=all_sha,
                                 deterministic_reset_receipt=deterministic)
    result = dict(held_labels_opened=False, source=collector_sha, order=ORDER, rows=2655, visits=14,
                  families=summaries, frozen_sha256=FROZEN_SHA, chain_sha256=file_sha(CHAIN),
                  relab_sha256=file_sha(RELAB), legacy_sinc_qualified=False, terminal_score_allowed=False,
                  recovered_fbl=restored, recovery_source=source)
    path = HERE / "qualification.json"
    with path.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    print("PASS recovered historical warm cohort", file_sha(path), flush=True)


if __name__ == "__main__":
    import sys

    collect(sys.argv[1])
