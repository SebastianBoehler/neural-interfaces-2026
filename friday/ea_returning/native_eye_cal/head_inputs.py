"""Authenticate both immutable source-only heads without reading any labels."""

import json

import numpy as np

from eye_cal_scope import CAL, DIMENSION, FIT_SOURCE, RIDGE_C, own_visit
from eye_scope import PEOPLE, pin, rows_sha, sha
from verify_joint import all_fits


HEAD_SOURCE = "779d882d60602e913ea9002ba5d82521a8f37487275011e425e4aaea2dc2833c"


def load():
    pin(CAL / "source.sha256", HEAD_SOURCE)
    receipt, bank = CAL / "heads.json", CAL / "heads.npz"
    proof = json.loads(receipt.read_text())
    fits, smoke = all_fits(FIT_SOURCE)
    scope = dict(source=HEAD_SOURCE, fits=fits, source_smoke=smoke, fit_source=FIT_SOURCE, classes=[0, 1, 2],
                 ridge_c=RIDGE_C, width=DIMENSION, scope="own_visit_1_only", eligible_people=list(PEOPLE),
                 order="full0|all_loso", later_visit_waveforms_used=False, later_visit_labels_opened=False)
    if (any(proof[k] != v for k, v in scope.items()) or set(proof["heads"]) != set(PEOPLE)
            or any(f.stat().st_mode & 0o222 for f in (receipt, bank))):
        raise ValueError("unauthenticated source-only Eye head proof")
    pin(bank, proof["output"])
    heads = {}
    with np.load(bank, allow_pickle=False) as z:
        for who in PEOPLE:
            rows = own_visit(who)
            inner = proof["heads"][who]
            np.testing.assert_array_equal(rows, z[f"{who}|rows"])
            if (inner["n"] != len(rows) or inner["rows"] != rows_sha(rows)
                    or not inner["exact_source_eye_prefix"] or not 0 <= inner["frozen_logits_max_abs"] <= 1e-5
                    or inner["fit_head_iterations"] >= 1000 or inner["fitted_norm"] <= .1):
                raise ValueError("unusable unseen source calibration")
            center, scale, coef, intercept = (z[f"{who}|{k}"] for k in ("center", "scale", "coef", "intercept"))
            if ((center.shape, scale.shape, coef.shape, intercept.shape) !=
                    ((DIMENSION,), (DIMENSION,), (3, DIMENSION), (3,))
                    or (scale < .005).any() or not all(np.isfinite(v).all() for v in (center, scale, coef, intercept))):
                raise ValueError("invalid calibrated inferential heads")
            for value in (center, scale, coef, intercept):
                value.setflags(write=False)
            heads[who] = (center, scale, coef, intercept)
    return fits, heads, sha(receipt)
