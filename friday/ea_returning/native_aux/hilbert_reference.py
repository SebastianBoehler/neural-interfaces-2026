"""Compute the old and fresh causal Hilbert references without held labels."""

import json
from pathlib import Path

import numpy as np
import ts_core2 as ts

from artifact import PILOT, PILOT_BANK, PILOT_SHA, REPO, pin, sha, source_params


CFG = "car_fb7_hfs2_t0.1_le_k2"


def causal(x, channels, p):
    n = len(x)
    data = dict(X=x, y=np.full(n, -1), ch_names=channels, stream=np.repeat("held-prefix", n))
    feats = ts.Feats(data, ts.parse(CFG))
    f = feats.causal(np.arange(n), p["log_glob"], None, None)
    if (f.shape != (n, 35629) or not np.isfinite(f).all() or (data["y"] != -1).any()):
        raise ValueError("label-blind Hilbert reference differs")
    return ts.softmax(((f - p["mu"]) / p["sd"]) @ p["W"].T + p["b"])


def main():
    new, fit, bank, held = source_params()
    pin(PILOT / "native_inner.json", PILOT_SHA)
    pin(PILOT / "native_inner.npz", PILOT_BANK)
    with np.load(PILOT / "native_inner.npz", allow_pickle=False) as z:
        rows = z["rows"]
    with np.load(REPO / "data/proteus_windows.npz", allow_pickle=False) as z:
        subjects, visit, channels = z["subject"], z["session"], z["ch_names"]
        first = np.flatnonzero((subjects == "Dreyer2026Proteus/P10") & (visit == "2"))[:31]
        raw = z["X"][np.r_[rows, first]]
    if rows.shape != (31,) or rows[0] != 1238 or not np.isin(rows, held).all():
        raise ValueError("reference opened the wrong held row scope")
    with np.load(REPO / "runs/sealed/cand_rec_ts2/weights.npz", allow_pickle=False) as z:
        old = {k: z["ts2_" + k] for k in new}
    prior = causal(raw[31:], channels, old)
    with np.load(REPO / "runs/sealed/cand_rec_ts2_harness.npz", allow_pickle=False) as z:
        reference = z["P_h"][first]
    error = float(np.abs(prior - reference).max())
    if error > 1e-6 or not np.array_equal(prior.argmax(1), reference.argmax(1)):
        raise ValueError(f"updated Hilbert harness did not reproduce historical selected path: {error}")
    a, b = [causal(raw[:31], channels, p) for p in (old, new)]
    for p, vals in ((old, a), (new, b)):
        prefix = causal(raw[:17], channels, p)
        if float(np.abs(prefix - vals[:17]).max()) > 1e-12:
            raise ValueError("reference is not a strict test prefix")
    path = Path("reference.npz")
    with path.open("xb") as handle:
        np.savez_compressed(handle, rows=rows, old=a, new=b)
    path.chmod(0o444)
    proof = dict(fit=fit, core_bank=bank, old_owning_path_max_abs=error, origin=PILOT_SHA, held_labels_opened=False,
                 source=sha("ref_source.sha256"), output=sha(path))
    meta = path.with_suffix(".json")
    with meta.open("x") as handle:
        json.dump(proof, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    meta.chmod(0o444)
    print("PASS independent causal Core2 references", sha(meta), flush=True)


if __name__ == "__main__":
    main()
