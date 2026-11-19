"""Pin the three authenticated source and held parents of the factorial."""
import json
from pathlib import Path

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, RUNTIME, row_sha, sha, split
from hierarchy_source import F17_ACTOR, F17_BANK, controls
from phase_source import F16_ACTOR, F16_BANK, authenticated
from temporal import BASE_PARAMS, CHANNELS, MODES, pair_sha


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "jitter", "lag", "joint")
ARMS = tuple(f"{kind}_{mode}" for kind in KINDS for mode in MODES)
F19_ACTOR = "4770494bdeb0cef1fd767d2359759657a72c54032faf7c26c62e0493c702496e"
F22_ACTOR = "8a7d1784adbd88fd7381e9f3628a613fa1b7768c4732ca2551ae37cb9687f2c1"
F22_BANK = "9c9dff817b519229033339f402e735191cd01b4a6e913c70df1db956093229e5"
F22_PINS = "c14a50a48d77ac0843adfaf3a115021991faf06c2b2e891f590ab5af61fa0010"
PARENTS = dict(hard="distill_20261118", jitter="phase_20261118", lag="temporal_v3_20261119")
PARENT_BANKS = dict(hard=F16_BANK, jitter=F17_BANK, lag=F22_BANK)


def parent_roots(independent):
    prefix = "ni26_friday_s1c_" if independent else "friday_s1c_"
    return {kind: HERE.parent / f"{prefix}{suffix}" for kind, suffix in PARENTS.items()}


def tag_for(held, seed, smoke):
    return f"{'smoke_' if smoke else ''}{seed}_{held}"


def operator(kind):
    return "lag" if kind in ("lag", "joint") else "hard"


def is_jittered(kind):
    return kind in ("jitter", "joint")


def parameter_count(kind):
    return BASE_PARAMS + CHANNELS * (operator(kind) == "lag")


def lag_reference(root, folder, tag, base):
    pins = HERE / "f22_lag.sha256"
    if sha(pins) != F22_PINS or sha(root / "actor.sha256") != F22_ACTOR:
        raise ValueError("F22 ancestor manifest differs")
    entries = {name: digest for digest, name in (line.split() for line in pins.read_text().splitlines())}
    stem = f"lag_{tag}"
    cp, receipt = folder / f"{stem}.pt", folder / f"{stem}.json"
    meta = json.loads(receipt.read_text())
    expected = {k: base[k] for k in ("held", "seed", "epochs", "scale", "origin", "runtime", "initial",
                                    "batches", "source_rows", "source_people", "steps")}
    expected.update(kind="lag", actor=F22_ACTOR, pair=pair_sha("lag"), data=DATA_SHA,
                    weights_origin=base["checkpoint"], checkpoint=entries[cp.name],
                    parameters=parameter_count("lag"), held_labels_opened=False, q_gradients_nonzero=0)
    if (sha(cp) != entries[cp.name] or sha(receipt) != entries[receipt.name] or
            any(meta[k] != v for k, v in expected.items())):
        raise ValueError(f"exact lag source parent differs for {tag}")
    return meta


def context(held, seed, smoke, subject, visit, independent):
    roots = parent_roots(independent)
    folders = {k: r / "checkpoints" if independent else r for k, r in roots.items()}
    rows, stream, people = split(subject, visit, held)
    base, batches = authenticated(roots["hard"], folders["hard"], held, seed, smoke, rows, stream, people)
    f17, same, directions, masks, tape = controls(roots["jitter"], folders["jitter"],
                                                 held, seed, smoke, rows, stream, people)
    lag = lag_reference(roots["lag"], folders["lag"], tag_for(held, seed, smoke), base)
    flat = [r for ep in batches for r in ep]
    if (f17["hard"]["checkpoint"] != base["checkpoint"] or any(not np.array_equal(a, b) for a, b in zip(
            flat, (r for ep in same for r in ep), strict=True)) or
            any(sha(roots[k] / "before_labels.sha256") != digest for k, digest in PARENT_BANKS.items() if independent)):
        raise ValueError("source/held exact parents or common batch schedule differ")
    return dict(hard=base, jitter=f17["jitter"], lag=lag), rows, stream, batches, directions, masks, tape


def common(refs, rows, batches, masks, tape):
    base = refs["hard"]
    first = batches[0][0][:32]
    return dict(held=base["held"], seed=base["seed"], epochs=base["epochs"], scale=base["scale"],
                initial=base["initial"], batches=base["batches"], source_rows=row_sha(rows),
                source_people=base["source_people"], actor=sha(HERE / "actor.sha256"), data=DATA_SHA,
                origin=ORIGIN_SHA, runtime=RUNTIME, train_rows_seen=sum(map(len, masks)),
                diagnostic_rows=row_sha(rows[first]), jitter_tape=tape,
                parents={k: m["checkpoint"] for k, m in refs.items()}, rng_isolated=True,
                zero_identity=True, held_labels_opened=False, torch=base["torch"], braindecode=base["braindecode"],
                q_gradients_nonzero=0)


def arm_fields(kind, masks):
    return dict(kind=kind, pair=pair_sha(operator(kind)), parameters=parameter_count(kind),
                perturbed_rows=sum(int(m.sum()) for m in masks) if is_jittered(kind) else 0,
                feed_calls=len(masks) if is_jittered(kind) else 0)
