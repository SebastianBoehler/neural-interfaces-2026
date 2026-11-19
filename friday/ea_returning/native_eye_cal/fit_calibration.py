"""Fit the fixed Eye heads on participant visit 1, unseen by the base actor."""

import json
from pathlib import Path
import runpy

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from eye_cal_scope import DIMENSION, FIT_SOURCE, RIDGE_C, own_visit
from eye_embedding import embeddings, load_networks
from eye_scope import DATA, OCUNET, PEOPLE, rows_sha, sha
from joint_eye import scope
from source_labels import read_training_labels
from verify_joint import all_fits


def fit_one(who, nets, source, excluded, total):
    rows = own_visit(who)
    if np.intersect1d(source, rows).size or not np.isin(rows, excluded).all():
        raise ValueError("the calibration rows entered the Eye feature producer fit")
    with np.load(DATA, allow_pickle=False) as d:
        fields = dict(X=d["X"][rows], ch_names=d["ch_names"], stream=np.zeros(len(rows), dtype=int))
    ocular = runpy.run_path(str(OCUNET))["traces"]
    features = ocular(fields)
    short = ocular(dict(fields, X=fields["X"][:17], stream=fields["stream"][:17]))
    np.testing.assert_array_equal(features[:17], short)
    e, bound = embeddings(nets, features, check=True)
    if e.shape != (len(rows), DIMENSION):
        raise ValueError("wrong complete source Eye representation")
    center, scale = e.mean(0), np.maximum(e.std(0), .005)
    index = np.argsort(rows)
    labels = read_training_labels(DATA, rows[index], total)[np.argsort(index)]
    actor = LogisticRegression(C=RIDGE_C, class_weight="balanced", solver="lbfgs", max_iter=1000, tol=1e-9)
    actor.fit((e - center) / scale, labels)
    np.testing.assert_array_equal(actor.classes_, (0, 1, 2))
    if (actor.coef_.shape != (3, DIMENSION) or np.linalg.norm(actor.coef_) <= .1
            or actor.n_iter_.max() >= 1000 or not np.isfinite(actor.coef_).all()):
        raise ValueError("inactive or unfit real source calibration head")
    pars = dict(center=center, scale=scale, coef=actor.coef_, intercept=actor.intercept_, rows=rows)
    metrics = dict(n=len(rows), rows=rows_sha(rows), fit_head_iterations=int(actor.n_iter_.max()),
                   fitted_norm=float(np.linalg.norm(actor.coef_)), frozen_logits_max_abs=bound,
                   exact_source_eye_prefix=True)
    return pars, metrics


def main():
    torch.set_num_threads(1)
    seed_records, smoke = all_fits(FIT_SOURCE)
    nets = load_networks(seed_records)
    train, held, people, _ = scope()
    arrays, proofs = {}, {}
    for who in PEOPLE:
        a, proofs[who] = fit_one(who, nets, train, held, len(people))
        arrays.update({f"{who}|{k}": v for k, v in a.items()})
    path = Path("heads.npz")
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)
    path.chmod(0o444)
    receipt = path.with_suffix(".json")
    proof = dict(source=sha("source.sha256"), fits=seed_records, source_smoke=smoke, fit_source=FIT_SOURCE,
                 classes=[0, 1, 2], ridge_c=RIDGE_C, width=DIMENSION, scope="own_visit_1_only",
                 eligible_people=PEOPLE, order="full0|all_loso", heads=proofs,
                 later_visit_waveforms_used=False, later_visit_labels_opened=False, output=sha(path))
    with receipt.open("x") as f:
        json.dump(proof, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    receipt.chmod(0o444)
    print("PASS authenticated calibration heads", sha(receipt), flush=True)


if __name__ == "__main__":
    main()
