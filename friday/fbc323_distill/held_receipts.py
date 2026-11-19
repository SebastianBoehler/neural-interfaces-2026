"""Independently verify each F16 source-only fold and its immutable artifacts."""
import json

import numpy as np

from anchors import DATA_SHA, HERE, KINDS, ORIGIN_SHA, RUNTIME, anchor, row_sha, schedule, sha
from source_targets import TOL, donor_plan


F15 = HERE.parent / "ni26_friday_s1c_skip_20261118"
F15_BANK = "92015e3dcc743708bb18f19ba090d5370645f3c184eeec5d635a5ccc83e54b7f"
ACTOR_SHA = "33d13bad6fe04b0ef14d8fb4f91f5034cf56a8d928ed38b3a09eca82a4c8c0d5"


def source_receipts(held, seed, smoke, rows, stream, roster, source_y=None):
    if sha(HERE / "actor.sha256") != ACTOR_SHA or sha(F15 / "before_labels.sha256") != F15_BANK:
        raise ValueError("source release or pinned F15 decisions differ")
    t_path, t = anchor(F15 / "checkpoints", "dropout", held, seed, False, rows, stream, roster)
    h_path, h = anchor(F15 / "checkpoints", "zero", held, seed, smoke, rows, stream, roster)
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    directory = HERE / "checkpoints"
    teacher = directory / f"teacher_{tag}.npz"
    plan = directory / f"plan_{tag}.npz"
    batches, trace = schedule(rows, stream, seed, h["epochs"])
    canonical = np.stack([rows[np.concatenate(epoch)] for epoch in batches])
    with np.load(teacher, allow_pickle=False) as z:
        logits, index = z["logits"], z["index"]
        receipt = json.loads(str(z["metadata"]))
    sessions = {g: row_sha(rows[stream == g]) for g in np.unique(stream)}
    proof = receipt["guards"]
    expected = dict(checkpoint=t["checkpoint"], receipt=sha(t_path.with_suffix(".json")),
                    source_rows=row_sha(rows), teacher_labels_opened=False)
    if (any(receipt[k] != v for k, v in expected.items()) or proof["sessions"] != sessions or
            proof["restarted"] != len(sessions) or not 0 <= proof["prefix_max_abs"] <= TOL or
            not np.array_equal(index, rows) or logits.shape != (len(rows), 3) or logits.dtype != np.float32 or
            not np.isfinite(logits).all()):
        raise ValueError("teacher source logits, chronology or exclusions differ")
    with np.load(plan, allow_pickle=False) as z:
        index, donors, groups = (z[key] for key in ("index", "donor", "groups"))
    if (not np.array_equal(index, canonical) or donors.shape != canonical.shape or
            not np.isin(donors, rows).all() or (index == donors).any() or
            groups.shape != (h["epochs"],) or (groups <= 0).any() or not np.array_equal(
                np.sort(index, axis=1), np.sort(donors, axis=1))):
        raise ValueError("source derangement changed padded coverage or reused physical rows")
    actual, control = np.searchsorted(rows, index), np.searchsorted(rows, donors)
    if not np.array_equal(stream[actual], stream[control]):
        raise ValueError("source derangement crossed session boundaries")
    if source_y is not None:
        _, expected_donors, expected_groups = donor_plan(batches, rows, stream, source_y, held, seed)
        if not np.array_equal(donors, expected_donors) or not np.array_equal(groups, expected_groups):
            raise ValueError("source derangement differs from the fixed within-class plan")

    common = dict(held=held, seed=seed, epochs=h["epochs"], scale=h["scale"], source_rows=row_sha(rows),
                  source_people=roster, dataset=DATA_SHA, origin=ORIGIN_SHA, actor=ACTOR_SHA, runtime=RUNTIME,
                  batches=trace, teacher=t["checkpoint"], teacher_bank=sha(teacher), donor_plan=sha(plan),
                  hard_origin=h["checkpoint"], hard_origin_receipt=sha(h_path.with_suffix(".json")),
                  held_labels_opened=False, source_only_labels=True, teacher_overlap="in_sample",
                  parameters=347903, initial=h["initial"], q_gradients_nonzero=0, temperature=2.,
                  steps=sum(map(len, batches)), braindecode=h["braindecode"], torch=h["torch"])
    result = {}
    for kind in KINDS:
        path = directory / f"{kind}_{tag}.pt"
        meta = json.loads(path.with_suffix(".json").read_text())
        expect = dict(common, checkpoint=sha(path), kind=kind, alpha=0. if kind == "hard" else .25)
        if (any(meta[k] != v for k, v in expect.items()) or "H100" not in meta["device"] or
                not np.isfinite([meta[k] for k in ("first_loss", "last_loss", "max_gradient", "max_target_delta")]).all()
                or meta["max_target_delta"] <= 1e-4 or meta["max_gradient"] <= 0):
            raise ValueError(f"student source fit provenance differs for {kind}/{tag}")
        if kind == "hard":
            if meta["state"] != h["state"]:
                raise ValueError("hard CE control no longer reproduces F15")
        elif not all(np.isfinite(meta[k]) and meta[k] > 0 for k in ("kl_body", "kl_head", "kl")):
            raise ValueError("source soft supervision did not reach the actual raw pathway")
        result[kind] = path, meta
    if len({m["state"] for _, m in result.values()}) != 3 or len(
            {m["device"] for _, m in result.values()}) != 1:
        raise ValueError("unpaired source runtime or duplicated training outcome")
    return result
