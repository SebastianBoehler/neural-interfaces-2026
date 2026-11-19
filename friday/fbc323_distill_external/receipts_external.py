"""Verify the all-source teachers, student states and padded donor coverage."""
import json

import numpy as np

from anchors import DATA_SHA, KINDS, ORIGIN_SHA, RUNTIME, row_sha, schedule, sha
from external_inputs import SOURCE, TOL
from external_source import F15E_BANK, HERE, all_source, teacher_receipt
from source_targets import donor_plan


F15E = HERE.parent / "ni26_friday_s1c_skip_external_20261118"
ACTOR = "b4dd2f3eb7ae50c417334fed30d4a5f06553e28631892245e0e456983601441a"


def paired(seed, smoke, source_labels=None):
    if sha(HERE / "actor.sha256") != ACTOR or sha(F15E / "before_labels.json") != F15E_BANK:
        raise ValueError("source actor or locked F15e ancestry differs")
    _, rows, stream, people = all_source(SOURCE, waveforms=False)
    parent, teacher = teacher_receipt(F15E, F15E / "checkpoints", rows, stream, people, seed)
    tag = f"{'smoke_' if smoke else ''}{seed}_all_source"
    directory = HERE / "checkpoints"
    bank, plan = (directory / f"{name}_{tag}.npz" for name in ("teacher", "plan"))
    with np.load(bank, allow_pickle=False) as z:
        logits, index, receipt = z["logits"], z["index"], json.loads(str(z["metadata"]))
    segments = {g: row_sha(rows[stream == g]) for g in np.unique(stream)}
    g = receipt["guards"]
    if (receipt["checkpoint"] != teacher["checkpoint"] or receipt["receipt"] != sha(parent.with_suffix(".json")) or
            receipt["source_rows"] != row_sha(rows) or receipt["teacher_labels_opened"] or
            g["sessions"] != segments or g["restarted"] != len(segments) or not 0 <= g["prefix_max_abs"] <= TOL or
            not np.array_equal(index, rows) or logits.shape != (len(rows), 3) or logits.dtype != np.float32 or
            not np.isfinite(logits).all()):
        raise ValueError("label-blind all-source teacher logits or boundaries differ")
    with np.load(plan, allow_pickle=False) as z:
        exposures, donors, groups = (z[k] for k in ("index", "donor", "groups"))
    epochs = 4 if smoke else 40
    batches, trace = schedule(rows, stream, seed, epochs)
    expected = np.stack([rows[np.concatenate(ep)] for ep in batches])
    if (not np.array_equal(exposures, expected) or donors.shape != exposures.shape or
            not np.isin(donors, rows).all() or (exposures == donors).any() or groups.shape != (epochs,) or
            not np.array_equal(np.sort(exposures, axis=1), np.sort(donors, axis=1)) or not (groups > 0).all() or
            not np.array_equal(stream[exposures], stream[donors])):
        raise ValueError("source donor coverage or original session differs")
    if source_labels is not None:
        _, checked_donors, checked_groups = donor_plan(batches, rows, stream, source_labels, "P00", seed)
        if not np.array_equal(donors, checked_donors) or not np.array_equal(groups, checked_groups):
            raise ValueError("source donors differ from the fixed class-preserving control")
    common = dict(held="all_source", seed=seed, epochs=epochs, scale=teacher["scale"], source_rows=row_sha(rows),
                  source_people=people, dataset=DATA_SHA, origin=ORIGIN_SHA, actor=ACTOR, runtime=RUNTIME, batches=trace,
                  teacher=teacher["checkpoint"], teacher_bank=sha(bank), donor_plan=sha(plan), initial=teacher["initial"],
                  parameters=347903, teacher_overlap="in_sample", held_labels_opened=False, source_only_labels=True,
                  torch=teacher["torch"], braindecode=teacher["braindecode"], temperature=2., q_gradients_nonzero=0,
                  steps=sum(map(len, batches)))
    result = {}
    for kind in KINDS:
        p = directory / f"{kind}_{tag}.pt"
        m = json.loads(p.with_suffix(".json").read_text())
        expect = dict(common, checkpoint=sha(p), kind=kind, alpha=0. if kind == "hard" else .25)
        if (any(m[k] != value for k, value in expect.items()) or "H100" not in m["device"] or
                not np.isfinite([m[k] for k in ("first_loss", "last_loss", "max_gradient", "max_target_delta")]).all() or
                m["max_gradient"] <= 0 or m["max_target_delta"] <= 1e-4 or m["state"] == m["initial"]):
            raise ValueError(f"matched all-source fit provenance differs for {kind} seed {seed}")
        if kind != "hard" and not all(np.isfinite(m[k]) and m[k] > 0 for k in ("kl_body", "kl_head", "kl")):
            raise ValueError("student KD control did not use the raw body and classifier")
        result[kind] = p, m
    if len({m["state"] for _, m in result.values()}) != 3 or len({m["device"] for _, m in result.values()}) != 1:
        raise ValueError("duplicate student states or unmatched training devices")
    return result
