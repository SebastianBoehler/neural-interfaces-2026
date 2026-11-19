"""Authenticate the fresh source Core2 without loading source or held labels."""

import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path("/mnt/lustre/work/gehler/gfh527")
REPO = BASE / "neural-interfaces-2026"
FIT = BASE / "ni26_friday_returning_core2_fit_v1_20261119"
FIT_SOURCE = "5b978c486a3e568838881d0334d6497148a9ace2a94745995076d6a4ba5fb638"
FOLD_HASH = "db255fd45af6d17fa4ba1829781bac5a92a08afa1bec4e5bf85266b56ea84790"
QUERY_HASH = "8aaa3c2c7d59a3851fb0c6cb0ece879771484a9a0151885c645df7b889e439b2"
PILOT = BASE / "ni26_friday_returning_native_inner_v2_20261119/smoke"
PILOT_SHA = "7574a7386c38a324b7bb0d65cf86df704a71f4a6944698cc2c16f45a7416f07c"
PILOT_BANK = "7b754c57959ac6305a58d6a9ce63873951cbbdfb48e37c59f61b7d4ad98a9f3c"
KEYS = ("bands", "k", "shrink", "slow_cutoff", "slow_step", "car", "nsub", "hilbert", "log_glob", "mu", "sd", "W", "b")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pin(path, expected):
    if sha(path) != expected:
        raise ValueError(f"artifact digest differs {path}")


def digest(rows):
    return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()


def source_params():
    pin(FIT / "fit_source.sha256", FIT_SOURCE)
    proof = FIT / "fit.json"
    bank = FIT / "core2.npz"
    rec = json.loads(proof.read_text())
    pins = dict(status="fresh_source_core2_fit", fold="P10_s23", cfg="car_fb7_hfs2_t0.1_le_k2", C=3.,
                source=FIT_SOURCE, train_rows=2270, held_rows=385, training_rows_sha256=FOLD_HASH,
                held_rows_sha256=QUERY_HASH, query_labels_read=False, query_features_used_in_fit=False)
    if any(rec[k] != v for k, v in pins.items()) or proof.stat().st_mode & 0o222 or bank.stat().st_mode & 0o222:
        raise ValueError("Hilbert fit provenance differs")
    pin(bank, rec["artifact_sha256"])
    with np.load(bank, allow_pickle=False) as z:
        train, held = z["fit_rows"], z["held_rows"]
        if digest(train) != FOLD_HASH or digest(held) != QUERY_HASH or len(np.intersect1d(train, held)):
            raise ValueError("Hilbert fit and held indices differ")
        params = {k: z[k] for k in KEYS}
    return params, sha(proof), rec["artifact_sha256"], held
