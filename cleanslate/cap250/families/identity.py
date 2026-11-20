"""Refit the identity gate and own-space stage (add_bio_gate.py, bio_gate.fold_params, add_own_stage.py)."""

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ASSETS, ROOT, family_args, load_source

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments"))
import _id_own
from bio_gate import Bio, fold_params

MODEL = "runs/replay_data/bio_model_psd_d64.npz"  # external two-covariance model (bio_id.py BIO_EXPORT, six external corpora)
TAU, RAMP, T0, BETA = .26, .05, 10, .1


def fit(args):
    data = load_source(args.people, args.first_session_only, args.data)
    names = list(args.people)  # sorted roster (source.roster)
    with np.load(ASSETS / MODEL, allow_pickle=False) as model_file:
        model = {key: model_file[key] for key in model_file.files}
    bio = Bio.__new__(Bio)
    bio.win = np.hanning(120)
    stream = np.char.add(np.char.add(data["person"], "|"), data["session"])
    feats = np.stack([bio.feat(x) for x in data["X"]])
    params = {key: np.asarray(value) for key, value in
              fold_params(model, feats, stream, names, tau=TAU, ramp=RAMP, t0=T0).items()}
    space = _id_own.Space(data["X"], data["y"])
    parts = []
    for name in names:
        rows = [np.flatnonzero((data["person"] == name) & (data["session"] == s))
                for s in sorted(set(data["session"][data["person"] == name]))]
        if not all(len(r) for r in rows):
            raise ValueError(f"empty source session for {name}")
        parts.append(space.person(rows))
    for key in _id_own.KEYS:
        params[f"own_{key}"] = np.stack([p[key] for p in parts]).astype(np.float64)
    params.update(own_bands=np.asarray(_id_own.BANDS, dtype=np.float64), own_beta=np.float64(BETA),
                  own_kref=np.float64(_id_own.KREF))
    if len(params["bio_gal"]) != len(names) or len(params["own_prior"]) != len(names):
        raise ValueError("identity per-person axis does not follow the roster")
    np.savez(args.out / "weights.npz", **params)


if __name__ == "__main__":
    fit(family_args(__doc__, (0,)))
