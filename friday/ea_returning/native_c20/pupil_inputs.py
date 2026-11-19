"""Authenticate the exact exported hard pupils and native delivery oracles."""

import json

import numpy as np

from eye_scope import DATA, DATA_SHA, ORDER_SHA, pin, rows_sha, sha
from pupil_source import ACTORS, PINS


SOURCE = "6a5f2ee2ef6f2fd8fbfcfcd536384892843a3535d6cb63433b3276cbc16be8d2"
TOL = 2e-5


def inputs(who, day, rows):
    pin(ACTORS / "source.sha256", SOURCE)
    pin(DATA, DATA_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        people = z["subject"]
    train = people != f"Dreyer2026Proteus/{who}"
    models, references, provenance = [], [], []
    for seed in (0, 1):
        path = ACTORS / who / f"hard_s{seed}.pt"
        rec, oracle = path.with_suffix(".json"), path.with_suffix(".npz")
        meta = json.loads(rec.read_text())
        fixed = dict(source=SOURCE, person=who, seed=seed, source_checkpoint=PINS[who, seed][0],
                     fit_rows=rows_sha(np.flatnonzero(train)), fit_people=sorted(set(people[train])),
                     order=ORDER_SHA, data=DATA_SHA, signed_all_loso_exact=True,
                     own_visit1_excluded=True, held_labels_opened=False, runtime="2.14.1+cu130")
        visit = meta["visits"][day]
        if (any(meta[k] != v for k, v in fixed.items()) or visit["delivery"] != rows_sha(rows)
                or visit["rows"] != len(rows) or not 0 <= visit["max_abs"] <= TOL
                or not all(visit[k] for k in ("exact_oracle_classes", "exact_prefix_reset", "exact_first"))
                or any(f.stat().st_mode & 0o222 for f in (path, rec, oracle))):
            raise ValueError("unqualified returning actual hard-CE pupil")
        pin(path, meta["artifact"])
        pin(oracle, meta["oracle"])
        with np.load(oracle, allow_pickle=False) as z:
            np.testing.assert_array_equal(z[f"rows{day}"], rows)
            prob = z[f"oracle{day}"]
        if prob.shape != (len(rows), 3) or not np.isfinite(prob).all() or (prob < 0).any():
            raise ValueError("invalid untouched eager returning oracle")
        np.testing.assert_allclose(prob.sum(1), 1, rtol=0, atol=1e-12)
        models.append(path)
        references.append(prob)
        provenance.append(dict(seed=seed, receipt=sha(rec), actor=meta["artifact"], checkpoint=meta["source_checkpoint"],
                               source_rows=meta["fit_rows"], oracle=meta["oracle"]))
    return models, np.stack(references, axis=1), provenance
