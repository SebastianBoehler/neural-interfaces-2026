"""Authenticate frozen source folds and their matched crossover diagnostics without gold."""
import json

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, row_sha, sha, split
from crossfade import HERE, KINDS, PARAMETERS, TOL
from group import ACTOR as F33A
from group import paired as anchor
from pairs import audit, plan
from recipe import BD, CPU_CUDA, GPU, TORCH, canary, historical, tag


ACTOR = "baf7735eb9ca7f225b99ecbf58e6a16a8fea3a13813ae7c2e8434d0fff901510"
F33 = HERE.parent / "ni26_friday_s1c_covariance_a100_whole_20261119"
F33_BLIND = "f996e3a55d376f6856bbd07425632033924a175636acc21b332d1a4d3f4f0525"
DIAGNOSTICS = (
    "severity_error", "cross_severity", "same_severity", "cross_seam", "same_seam",
    "treatment_movement", "cross_fraction", "same_fraction",
)


def pairing(held, seed, smoke, subject, visit, labels=None):
    rows, streams, people = split(subject, visit, held)
    ref, batches, _ = historical(held, seed, smoke, rows, streams, people)
    path = HERE / "checkpoints" / f"pairs_{tag(held, seed, smoke)}.npz"
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"pairs", "metadata"}:
            raise ValueError("source tape schema differs")
        tape, meta = z["pairs"], json.loads(str(z["metadata"]))
    source_labels = None if labels is None else labels[rows]
    _, proof = audit(batches, streams, tape, source_labels)
    if source_labels is not None and not np.array_equal(tape, plan(batches, streams, source_labels, held, seed)):
        raise ValueError("source labels or deterministic matched crossover tape differ")
    return ref, rows, people, meta, proof, sha(path)


def paired(held, seed, smoke, subject, visit):
    if sha(HERE / "actor.sha256") != ACTOR or sha(F33 / "before_labels.sha256") != F33_BLIND:
        raise ValueError("F36 source or A100 held ancestry differs")
    ref, rows, people, meta, proof, lock = pairing(held, seed, smoke, subject, visit)
    native = anchor(held, seed, smoke, subject, visit)["hard"][1]
    env = canary()
    fixed = dict(
        held=held, seed=seed, epochs=ref["epochs"], scale=native["scale"], initial=ref["initial"],
        data=DATA_SHA, origin=ORIGIN_SHA, source_rows=row_sha(rows), source_people=people,
        batches=ref["batches"], held_labels_opened=False, actor=ACTOR, f33a=F33A,
        weights_origin=native["checkpoint"], device=GPU, torch=TORCH, braindecode=BD,
        parameters=PARAMETERS, q_gradients_nonzero=0, rng_isolated=True,
        **{k: env[k] for k in CPU_CUDA}, **proof,
    )
    if meta != fixed:
        raise ValueError("source tape fixed recipe or row proof differs")
    result = {}
    for kind in KINDS:
        path = HERE / "checkpoints" / f"{kind}_{tag(held, seed, smoke)}.pt"
        m = json.loads(path.with_suffix(".json").read_text())
        expected = dict(fixed, kind=kind, checkpoint=sha(path), steps=ref["steps"], tape=lock)
        if (any(m[k] != v for k, v in expected.items()) or m["state"] == m["initial"] or
                not np.isfinite([m[k] for k in ("first_loss", "last_loss", "max_gradient")]).all() or
                m["max_gradient"] <= 0):
            raise ValueError(f"fit identity or source exposure differs for {kind}")
        if kind == "hard":
            if any(m[k] != native[k] for k in ("checkpoint", "state", "first_loss", "last_loss", "max_gradient")):
                raise ValueError("native hard A100 state or trajectory differs")
        else:
            if (m["steps_applied"] != ref["steps"] or m["changed_source_rows"] != proof["treated_rows"] or
                    m["zero_dose_exact"] is not True or not np.isfinite([m[k] for k in DIAGNOSTICS]).all() or
                    not 0 <= m["severity_error"] <= TOL or
                    abs(m["cross_severity"] - m["same_severity"]) > TOL or
                    not all(0 < m[k] <= .5 + TOL for k in ("cross_severity", "same_severity")) or
                    not all(0 < m[k] <= 1 for k in ("cross_fraction", "same_fraction")) or
                    min(m["cross_seam"], m["same_seam"]) < 0 or m["treatment_movement"] <= 1e-4):
                raise ValueError(f"exact raw dose, zero-input identity or crossover activity differs for {kind}")
        result[kind] = path, m
    a, b = (result[k][1] for k in ("cross_trial", "same_trial"))
    if any(a[k] != b[k] for k in DIAGNOSTICS) or len({m["state"] for _, m in result.values()}) != len(KINDS):
        raise ValueError("active perturbation diagnostics differed or fitted states were duplicated")
    return result
