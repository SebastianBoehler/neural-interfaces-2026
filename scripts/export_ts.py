"""Build cand_rec_ts (6 Nov): a sealed package with its tangent member (v2) replaced by the new causal TS core.

Usage: python scripts/export_ts.py SRC_DIR OUT_DIR [--cfg car_ov17_fs2_t0.1_le_k2] [--C 10] [--weight 1.5]
       [--old-labels SRC_harness.npz]
The core (experiments/ts_core.py, config naming there; head StandardScaler + LogisticRegression C) is trained on
all public windows exactly as ts_core's Core.fit (whole-session log-Euclidean recentering of the training
sessions, prior = log-Euclidean mean of all training windows) and exported as submission.py AlignedTangent keys
(ts_car, ts_nsub new; docs/AUDIT_SBN.md). member_weights[0] becomes weight x SRC's (v2 = 1 unit); every other key
and file of SRC is unchanged; submission.py is the current code.
Checks on all public windows, streamed per session: the package's tangent member equals ts_core's causal stream
(Core.predict_proba_stream, in-sample); the package's labels equal the harness (SRC's mix without the tangent
member + core probabilities -> rebalancing -> session EM, scripts/set_params.harness). Writes OUT_DIR_check.json
and OUT_DIR_harness.npz. --old-labels takes SRC's labels (key new) from its export instead of streaming SRC.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import session_em  # noqa: E402
import ts_core  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from set_params import harness  # noqa: E402

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cfg", default="car_ov17_fs2_t0.1_le_k2")
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--weight", type=float, default=1.5)
    ap.add_argument("--old-labels", type=Path)
    a = ap.parse_args()
    cfg = ts_core.parse(a.cfg)
    assert cfg["ref"] in ("car", "m1") and cfg["refm"] == "le" and not cfg["st"] and cfg["est"][0] == "t", cfg
    assert cfg["cov"] == "full" or cfg["cov"][:2] == "fs", cfg  # package: full window (+ nsub sub-windows)
    nsub = int(cfg["cov"][2:]) if cfg["cov"][:2] == "fs" else 0
    with np.load(a.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    w = z["member_weights"].astype(np.float64)

    d = protocol.load()
    st, rows, N = d["stream"], np.ones(len(d["y"]), bool), len(d["y"])
    res = {"source": str(a.src), "source_weights_sha256": sha(a.src / "weights.npz"), "cfg": a.cfg, "C": a.C,
           "weight_v2_units": a.weight, "ts_core_md5": hashlib.md5((ROOT / "experiments/ts_core.py").read_bytes()).hexdigest(),
           "git_rev": os.environ.get("NI26_SOURCE_REV"), "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID")}

    # train the core on all public windows; in-sample causal stream per session (ts_core.Core)
    t0 = time.time()
    fe = ts_core.Feats(d, cfg)
    head = f"lr{a.C:g}"
    store = {head: np.full((N, 3), np.nan)}
    core = ts_core.Core(fe, [head], store)
    core.fit(np.arange(N))
    for s in dict.fromkeys(st):
        core.predict_proba_stream(np.flatnonzero(st == s))
    P_ts = store[head]
    assert np.isfinite(P_ts).all()
    pipe = core.models[0]
    sc, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    assert list(lr.classes_) == [0, 1, 2], lr.classes_
    res["fit_seconds"], res["dim"] = round(time.time() - t0, 1), int(fe.F.shape[1])
    print("core fit", res["fit_seconds"], "s, dim", res["dim"], flush=True)

    # package keys (all ts_* keys replaced)
    ts = {"bands": np.array(ts_core.BANDSETS[cfg["bands"]], dtype=np.float64),
          "log_glob": np.stack(core.priors), "k": np.array(cfg["k"]), "shrink": np.array(float(cfg["est"][1:])),
          "mu": sc.mean_, "sd": sc.scale_, "W": lr.coef_, "b": lr.intercept_,
          "slow_cutoff": np.array(1.0), "slow_step": np.array(60), "car": np.array(int(cfg["ref"] == "car")),
          "nsub": np.array(nsub)}
    assert ts["log_glob"].shape[0] == len(ts["bands"]) * (1 + nsub) and ts["W"].shape[1] == res["dim"]
    zn = {k: v for k, v in z.items() if not k.startswith("ts_")}
    zn.update({f"ts_{k}": v for k, v in ts.items()})
    w_new = w.copy()
    w_new[0] = a.weight * w[0]
    zn["member_weights"] = w_new
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "weights.npz", **zn)
    for f in a.src.iterdir():
        if f.is_file() and f.name not in ("weights.npz", "submission.py"):
            shutil.copyfile(f, a.out / f.name)
    shutil.copyfile(SRC, a.out / "submission.py")

    # the package's tangent member == ts_core's causal stream
    tang = load_solver(a.out, "new_tangent").tangent
    t0 = time.time()
    P_pkg = stream(tang, d, rows, lambda m, x: m.proba(x[0])[None])
    res["tangent_seconds_per_window"] = round((time.time() - t0) / N, 5)
    res["tangent_max_abs_prob_diff"] = float(np.abs(P_pkg - P_ts).max())
    res["tangent_labels_equal"] = float(np.mean(P_pkg.argmax(1) == P_ts.argmax(1)))
    print("tangent", res["tangent_max_abs_prob_diff"], res["tangent_labels_equal"],
          res["tangent_seconds_per_window"], "s/window", flush=True)
    assert res["tangent_max_abs_prob_diff"] < 1e-4, res

    # harness: SRC's mix with the tangent member at weight 0 + core probabilities -> rebalance -> EM
    with tempfile.TemporaryDirectory() as tmp:
        rest = Path(tmp)
        for f in a.src.iterdir():
            if f.is_file() and f.name != "weights.npz":
                shutil.copyfile(f, rest / f.name)
        np.savez(rest / "weights.npz", **dict(z, member_weights=np.r_[0.0, w[1:]]))
        mix_rest = stream(load_solver(rest, "rest_mix"), d, rows, lambda mm, x: mm.mixed_proba(x))
    w_rest = float(w[1:].sum())
    mix = (mix_rest * w_rest + w_new[0] * P_ts) / (w_rest + w_new[0])
    X64 = d["X"].astype(np.float64)
    Fe = np.c_[session_em.gaze_features(X64, list(d["ch_names"])),
               session_em.event_features(X64, list(d["ch_names"]), st, z["em_ev_g"])]
    ref = harness(zn, mix, st, Fe)
    t0 = time.time()
    new = stream(load_solver(a.out, "new_pkg"), d, rows, lambda mm, x: mm.predict(x))
    res["package_cpu_seconds_per_window"] = round((time.time() - t0) / N, 4)
    if a.old_labels:
        old = np.load(a.old_labels)["new"]
        res["old_labels_from"] = str(a.old_labels)
    else:
        old = stream(load_solver(a.src, "src_pkg"), d, rows, lambda mm, x: mm.predict(x))
    score = lambda lab: protocol.score(d["y"], lab, d, rows)["balanced_accuracy_cell_mean"]
    res.update({"n_windows": int(N), "member_weights_0": [float(w[0]), float(w_new[0])],
                "member_weights_sum": float(w_new.sum()),
                "package_matches_harness": float(np.mean(new == ref)),
                "n_windows_differ_from_harness": int(np.sum(new != ref)),
                "labels_changed_vs_source": int(np.sum(new != old)),
                "in_sample_cell_mean": {"source": score(old), "package": score(new), "core_alone": score(P_ts.argmax(1))},
                "files_sha256": {f.name: sha(f) for f in sorted(a.out.iterdir()) if f.is_file()}})
    np.savez(a.out.parent / f"{a.out.name}_harness.npz", mix_rest=mix_rest, mix=mix, ref=ref, new=new, old=old,
             P_ts=P_ts, P_pkg=P_pkg)
    (a.out.parent / f"{a.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["package_matches_harness"] == 1.0


if __name__ == "__main__":
    main()
