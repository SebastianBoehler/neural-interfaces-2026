"""Fit the archived Hilbert Core2 recipe on the P10 §321 source rows."""

import hashlib
import json
from pathlib import Path
import resource
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits

from source_labels import read_training_labels
import ts_core2 as ts


BASE = Path("/mnt/lustre/work/gehler/gfh527")
REPO = BASE / "neural-interfaces-2026"
DATA = REPO / "data/proteus_windows.npz"
EXPORT = REPO / "runs/sealed/cand_rec_ts2_check.json"
PKG = BASE / "ni26_js321_20261117_returning_bundle2/P10_s23/base"
FIT = BASE / "ni26_js321_20261117_returning_view_fit_p10_1/source_view_P10_s23.npz"
CF = "car_fb7_hfs2_t0.1_le_k2"
DIM = 35629
HELD = f"Dreyer2026Proteus/P10"
FIT_SHA = "db255fd45af6d17fa4ba1829781bac5a92a08afa1bec4e5bf85266b56ea84790"
HELD_SHA = "8aaa3c2c7d59a3851fb0c6cb0ece879771484a9a0151885c645df7b889e439b2"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def row_hash(rows):
    return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()


def main():
    start = monotonic()
    original = json.loads(EXPORT.read_text())
    if (original["cfg"], original["C"], original["dim"], original["n_windows"]) != (CF, 3., DIM, 2655):
        raise ValueError("the owning all-public Hilbert recipe differs")
    with np.load(DATA, allow_pickle=False) as z:
        people, visit = z["subject"], z["session"]
        keep = (people != HELD) | (visit == "1")
        rows = np.flatnonzero(keep)
        held = np.flatnonzero(~keep)
        x, channels = z["X"][rows], z["ch_names"]
    if (len(rows), len(held), int(((people == HELD) & (visit == "1")).sum())) != (2270, 385, 192):
        raise ValueError("returning protocol scope differs")
    if row_hash(rows) != FIT_SHA or row_hash(held) != HELD_SHA:
        raise ValueError("the fitted source rows differ from §321")
    with np.load(FIT, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["idx_tr"], rows)
    data = dict(X=x, y=read_training_labels(DATA, rows, len(people)),
                stream=np.char.add(np.char.add(people[rows], "|"), visit[rows]), ch_names=channels)
    cfg = ts.parse(CF)
    common = dict(bands=np.asarray(ts.BANDSETS[cfg["bands"]], dtype=np.float64), k=np.array(cfg["k"]),
                  shrink=np.array(float(cfg["est"][1:])), slow_cutoff=np.array(1.), slow_step=np.array(60),
                  car=np.array(1), nsub=np.array(2), hilbert=np.array(1))
    with np.load(PKG / "weights.npz", allow_pickle=False) as native, np.load(
            REPO / "runs/sealed/cand_rec_ts2/weights.npz", allow_pickle=False) as orig:
        keys = {k for k in native.files if k.startswith("ts2_")}
        if keys != {f"ts2_{k}" for k in (*common, "log_glob", "mu", "sd", "W", "b")}:
            raise ValueError("the native Hilbert slot differs in schema")
        for k in keys:
            np.testing.assert_array_equal(native[k], orig[k])
        for k, v in common.items():
            np.testing.assert_array_equal(native[f"ts2_{k}"], v)
    print("P10 genuine source fit", len(rows), len(held), flush=True)
    with threadpool_limits(limits=1):
        feats = ts.Feats(data, cfg)
    if feats.F.shape != (2270, DIM) or feats.y is not data["y"]:
        raise ValueError("source Hilbert feature shape or row mapping differs")
    core = ts.Core(feats, ["lr3"], {"lr3": np.full((len(rows), 3), np.nan)})
    with threadpool_limits(limits=6):
        core.fit(np.arange(len(rows)))
    sc = core.models[0].named_steps["standardscaler"]
    lr = core.models[0].named_steps["logisticregression"]
    np.testing.assert_array_equal(lr.classes_, (0, 1, 2))
    params = dict(common, log_glob=np.stack(core.priors), mu=sc.mean_, sd=sc.scale_, W=lr.coef_, b=lr.intercept_)
    if (params["log_glob"].shape != (21, 41, 41) or not np.iscomplexobj(params["log_glob"])
            or params["W"].shape != (3, DIM) or (params["sd"] <= 0).any()
            or not all(np.isfinite(a).all() for a in params.values())):
        raise ValueError("invalid fresh Hilbert core")
    path = Path("core2.npz")
    with path.open("xb") as handle:
        np.savez_compressed(handle, fit_rows=rows, held_rows=held, **params)
    path.chmod(0o444)
    proof = dict(status="fresh_source_core2_fit", fold="P10_s23", cfg=CF, C=3., train_rows=len(rows), held_rows=len(held),
                 source=sha("fit_source.sha256"), data=sha(DATA), basis=sha(FIT), code=sha(ts.__file__),
                 original_export=sha(EXPORT), training_rows_sha256=FIT_SHA, held_rows_sha256=HELD_SHA,
                 artifact_sha256=sha(path), query_labels_read=False, query_features_used_in_fit=False,
                 elapsed=monotonic() - start, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    receipt = Path("fit.json")
    with receipt.open("x") as handle:
        json.dump(proof, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    receipt.chmod(0o444)
    print("PASS fitted Hilbert Core2", sha(receipt), flush=True)


if __name__ == "__main__":
    main()
