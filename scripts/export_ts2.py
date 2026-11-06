"""Build cand_rec_ts2 (6 Nov): cand_rec_ts plus a second tangent-space member on the bands' analytic signal.

Usage: python scripts/export_ts2.py SRC_DIR OUT_DIR --src-harness SRC_harness.npz [--cfg car_fb7_hfs2_t0.1_le_k2]
       [--C 3] [--w1 0.125] [--w2 0.125] [--code submissions/ensemble/submission.py]
SRC_DIR: a package whose tangent member is a 6 Nov core (scripts/export_ts.py, e.g. cand_rec_ts); SRC_harness.npz:
that export's harness (mix_rest = the nets' mix, P_ts = the core's in-sample stream, new = SRC's labels).
The Hilbert core (experiments/ts_core2.py, the verbatim dev harness, config naming there; head StandardScaler +
LogisticRegression C) is trained on all public windows exactly as ts_core2's Core.fit (training sessions recentred
by their whole-session log-Euclidean mean, prior = log-Euclidean mean of all windows) and exported as ts2_* keys:
a second AlignedTangent (ts2_hilbert 1). member_weights[0] (SRC's core) becomes w1 and w2 is appended for the new
member; every other key and file of SRC is unchanged; submission.py is --code.
Checks on all public windows, streamed per session: the package's second member equals ts_core2's causal stream
(features and probabilities), its first member equals SRC's (P_pkg), and its labels equal the harness (nets' mix +
both cores mixed with the package weights -> rebalancing -> session EM, scripts/set_params.harness).
Writes OUT_DIR_check.json and OUT_DIR_harness.npz.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import session_em  # noqa: E402
import ts_core2  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from set_params import harness  # noqa: E402

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()


def member_stream(m, d):
    """Features and probabilities of an AlignedTangent, window by window, reset per session."""
    F, P, st = [], [], d["stream"]
    for j in range(len(st)):
        if j == 0 or st[j] != st[j - 1]:
            m.reset_state()
        f = m._features(d["X"][j])
        F.append(f)
        z = m.p["W"] @ ((f - m.p["mu"]) / m.p["sd"]) + m.p["b"]  # = AlignedTangent.proba
        P.append(np.exp(z - z.max()) / np.exp(z - z.max()).sum())
    return np.array(F), np.array(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src-harness", type=Path, required=True)
    ap.add_argument("--cfg", default="car_fb7_hfs2_t0.1_le_k2")
    ap.add_argument("--C", type=float, default=3.0)
    ap.add_argument("--w1", type=float, default=0.125)
    ap.add_argument("--w2", type=float, default=0.125)
    ap.add_argument("--code", type=Path, default=SRC)
    a = ap.parse_args()
    cfg = ts_core2.parse(a.cfg)
    pre, lay = re.fullmatch(r"(h)(fs\d)", cfg["cov"]).groups()  # package: analytic signal, full + nsub windows
    assert cfg["ref"] == "car" and cfg["refm"] == "le" and cfg["est"][0] == "t" and "+" not in cfg["bands"], cfg
    assert not (cfg["st"] or cfg["sz"] or cfg["ct"] or cfg["cb"]), cfg
    nsub = int(lay[2:])
    with np.load(a.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    assert not any(k.startswith("ts2_") for k in z) and int(z["ts_nsub"]) > 0
    w = z["member_weights"].astype(np.float64)
    assert len(w) == 1 + int(z["n_nets"])
    Hs = dict(np.load(a.src_harness))

    d = protocol.load()
    st, rows, N = d["stream"], np.ones(len(d["y"]), bool), len(d["y"])
    res = {"source": str(a.src), "source_weights_sha256": sha(a.src / "weights.npz"),
           "source_harness_sha256": sha(a.src_harness), "cfg": a.cfg, "C": a.C, "w1": a.w1, "w2": a.w2,
           "code_sha256": sha(a.code), "ts_core2_md5": hashlib.md5((ROOT / "experiments/ts_core2.py").read_bytes()).hexdigest(),
           "git_rev": os.environ.get("NI26_SOURCE_REV"), "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID")}

    # train the Hilbert core on all public windows; in-sample causal stream per session (ts_core2.Core)
    t0 = time.time()
    fe = ts_core2.Feats(d, cfg)
    head = f"lr{a.C:g}"
    store = {head: np.full((N, 3), np.nan)}
    core = ts_core2.Core(fe, [head], store)
    core.fit(np.arange(N))
    res["fit_seconds"], res["dim"] = round(time.time() - t0, 1), int(fe.F.shape[1])
    print("core fit", res["fit_seconds"], "s, dim", res["dim"], flush=True)
    F_ref = np.empty_like(fe.F)
    for s in dict.fromkeys(st):
        r = np.flatnonzero(st == s)
        F_ref[r] = fe.causal(r, core.priors, core.d2p, core.szp)
        core.predict_proba_stream(r)
    P_h = store[head]
    assert np.isfinite(P_h).all()
    sc, lr = core.models[0].named_steps["standardscaler"], core.models[0].named_steps["logisticregression"]
    assert list(lr.classes_) == [0, 1, 2], lr.classes_
    del fe

    ts2 = {"bands": np.array(ts_core2.BANDSETS[cfg["bands"]], dtype=np.float64),
           "log_glob": np.stack(core.priors), "k": np.array(cfg["k"]), "shrink": np.array(float(cfg["est"][1:])),
           "mu": sc.mean_, "sd": sc.scale_, "W": lr.coef_, "b": lr.intercept_,
           "slow_cutoff": np.array(1.0), "slow_step": np.array(60), "car": np.array(1), "nsub": np.array(nsub),
           "hilbert": np.array(1)}
    assert ts2["log_glob"].shape[0] == len(ts2["bands"]) * (1 + nsub) and np.iscomplexobj(ts2["log_glob"])
    assert ts2["W"].shape[1] == res["dim"]
    w_new = np.r_[a.w1, w[1:], a.w2]
    zn = {**z, "member_weights": w_new, **{f"ts2_{k}": v for k, v in ts2.items()}}
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "weights.npz", **zn)
    for f in a.src.iterdir():
        if f.is_file() and f.name not in ("weights.npz", "submission.py"):
            shutil.copyfile(f, a.out / f.name)
    shutil.copyfile(a.code, a.out / "submission.py")

    # the package's members == the cores' causal streams
    m = load_solver(a.out, "new_pkg")
    t0 = time.time()
    F_pkg, P_pkg2 = member_stream(m.tangent2, d)
    res["tangent2_seconds_per_window"] = round((time.time() - t0) / N, 5)
    dF = np.abs(F_pkg - F_ref)
    res["tangent2_max_abs_feature_diff"] = float(dF.max())
    res["tangent2_max_standardised_feature_diff"] = float((dF / sc.scale_).max())
    res["tangent2_max_abs_prob_diff"] = float(np.abs(P_pkg2 - P_h).max())
    res["tangent2_labels_equal"] = float(np.mean(P_pkg2.argmax(1) == P_h.argmax(1)))
    del F_pkg, F_ref, dF
    P_pkg1 = stream(m.tangent, d, rows, lambda mm, x: mm.proba(x[0])[None])
    res["tangent1_max_abs_prob_diff_vs_source"] = float(np.abs(P_pkg1 - Hs["P_pkg"]).max())
    print(json.dumps(res, indent=1), flush=True)
    assert res["tangent2_max_abs_prob_diff"] < 1e-4 and res["tangent1_max_abs_prob_diff_vs_source"] < 1e-9, res

    # harness: the nets' mix (SRC export) + both cores, package weights -> rebalance -> EM
    w_rest = float(w[1:].sum())
    mix = (Hs["mix_rest"] * w_rest + a.w1 * Hs["P_ts"] + a.w2 * P_h) / (w_rest + a.w1 + a.w2)
    X64 = d["X"].astype(np.float64)
    Fe = np.c_[session_em.gaze_features(X64, list(d["ch_names"])),
               session_em.event_features(X64, list(d["ch_names"]), st, z["em_ev_g"])]
    ref = harness(zn, mix, st, Fe)
    t0 = time.time()
    new = stream(m, d, rows, lambda mm, x: mm.predict(x))
    res["package_cpu_seconds_per_window"] = round((time.time() - t0) / N, 4)
    old = Hs["new"]
    score = lambda lab: protocol.score(d["y"], lab, d, rows)["balanced_accuracy_cell_mean"]
    res.update({"n_windows": int(N), "member_weights_0": [float(w[0]), float(w_new[0])],
                "member_weights_last": float(w_new[-1]), "member_weights_sum": float(w_new.sum()),
                "package_matches_harness": float(np.mean(new == ref)),
                "n_windows_differ_from_harness": int(np.sum(new != ref)),
                "labels_changed_vs_source": int(np.sum(new != old)),
                "in_sample_cell_mean": {"source": score(old), "package": score(new), "core2_alone": score(P_h.argmax(1))},
                "files_sha256": {f.name: sha(f) for f in sorted(a.out.iterdir()) if f.is_file()}})
    np.savez(a.out.parent / f"{a.out.name}_harness.npz", mix=mix, ref=ref, new=new, old=old, P_h=P_h,
             P_pkg2=P_pkg2, P_pkg1=P_pkg1)
    (a.out.parent / f"{a.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["package_matches_harness"] == 1.0


if __name__ == "__main__":
    main()
