"""Verify each nested training scope and assemble the immutable source teacher matrix."""
import json

import numpy as np

from anchors import DATA_SHA, ORIGIN_SHA, RUNTIME, row_sha, schedule, sha
from relational import DIM, F25_ACTOR, HERE, TOL, groups, tag_for, teacher_rows


TEACHER_SHA = "467fd886eb03bc1b27ea28e42b199c89eb9d314630f584adf71b6e2db232354f"


def facts(person, stream, held, seed, group, smoke, base):
    fit, query = teacher_rows(person, held, group)
    batches, tape = schedule(fit, stream[fit], seed, base["epochs"])
    expected = dict(kind="multi", held=held, seed=seed, group=group, epochs=base["epochs"], actor=TEACHER_SHA,
                    data=DATA_SHA, f25=F25_ACTOR, runtime=RUNTIME, origin=ORIGIN_SHA, initial=base["initial"],
                    source_people=sorted(set(person[fit])), query_people=sorted(groups(held)[group]),
                    source_rows=row_sha(fit), query_rows=row_sha(query), batches=tape, steps=sum(map(len,batches)),
                    teacher_query_labels_opened=False, held_labels_opened=False, q_gradients_nonzero=0)
    return query, expected


def receipt(folder, person, stream, held, seed, group, smoke, base):
    query, expected = facts(person, stream, held, seed, group, smoke, base)
    stem = folder / f"teacher_{tag_for(held, seed, smoke)}_{group}"
    side = json.loads(stem.with_suffix(".seal").read_text())
    actual = json.loads(stem.with_suffix(".json").read_text())
    resets = len(set(stream[query]))
    if (set(side) != {"teacher", "receipt", "bank"} or any(len(v)!=64 for v in side.values()) or
            side["receipt"] != sha(stem.with_suffix(".json")) or side["teacher"] != actual["checkpoint"] or
            any(actual[k] != v for k,v in expected.items()) or "H100" not in actual["device"] or
            actual["restarts"] != resets or not 0 <= actual["prefix"] <= TOL or
            not 0 < actual["scale"] or not np.isfinite(actual["scale"]) or
            any(not np.isfinite(actual[k]) or actual[k] <= 0 for k in (
                "band_grad","tap_grad","cross_band","dilation_delta","identity_delta","edge_tap"))):
        raise ValueError(f"authentic teacher or six-person exclusion differs: {stem.name}")
    return stem, query, side, actual


def load(source, person, stream, held, seed, smoke, base):
    if sha(HERE/"teacher.sha256") != TEACHER_SHA:
        raise ValueError("teacher source differs")
    full = np.full((len(person), DIM), np.nan, dtype=np.float32)
    stamps = {}
    for group in range(3):
        stem, rows, side, actual = receipt(HERE/"teachers", person, stream, held, seed, group, smoke, base)
        if sha(stem.with_suffix(".pt")) != side["teacher"] or sha(stem.with_suffix(".npz")) != side["bank"]:
            raise ValueError("teacher checkpoint or frozen representation bytes differ")
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
            embedding = z["embedding"]
            unsigned = {k:v for k,v in actual.items() if k != "checkpoint"}
            if (not np.array_equal(z["index"], rows) or json.loads(str(z["meta"])) != unsigned or
                    embedding.shape != (len(rows),DIM) or embedding.dtype != np.float32 or
                    not np.isnan(full[rows]).all() or not np.isfinite(embedding).all() or
                    np.max(abs(np.linalg.norm(embedding,axis=1)-1)) > TOL):
                raise ValueError("teacher source-only embedding bank differs")
            full[rows] = embedding
        stamps[str(group)] = side
    if not np.array_equal(np.flatnonzero(np.isfinite(full).all(1)), source):
        raise ValueError("all and only outer source rows require a nested teacher")
    return full[source], stamps
