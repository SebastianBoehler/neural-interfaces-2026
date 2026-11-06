"""Build cand_rec_ts2_tsem (6 Nov): cand_rec_ts2 plus the TSEM stage after the gaze/event session EM.

Usage: python scripts/export_tsem.py SRC_DIR OUT_DIR --src-harness SRC_harness.npz [--npc 128] [--lam 0.15] [--rho 0.7]
       [--shrink 0.7] [--m0 20] [--n0 60] [--nmin 15] [--code submissions/ensemble/submission.py]
SRC_DIR: a package whose first tangent member is the 6 Nov core car_ov17_fs2_t0.1_le_k2 (e.g. cand_rec_ts2);
SRC_harness.npz: its export's harness (mix = pre-rebalancing member mix, new = SRC's labels).
TSEM view (experiments/tsem.py, batch reference): u_t = (z_t - m) @ V, z_t the core's scaled causal tangent features
(what its LR sees), V the top npc right singular vectors (exact gesdd SVD, svd_flip) of the centred scaled
whole-session training features of all public windows (ts_core.Feats rows, scaled by the packaged ts_mu / ts_sd).
Package: SRC + tsem_* keys (tsem_V float32); every other key and file unchanged; submission.py is --code.
Checks on all public windows, streamed per session: the packaged scaler / priors equal a fresh fit; the package's
views equal the reference views; its session-EM output equals set_params.harness's; its labels equal batch TSEM on
the harness posteriors with the packaged (float32) basis (and, informatively, with the float64 basis). Then, on
two sessions fed in batches of 64: truncated streams are bit-identical before T, and a deep-copied, reset model
equals a fresh one. Writes OUT_DIR_check.json and OUT_DIR_harness.npz.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import svd_flip

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import session_em  # noqa: E402
import ts_core  # noqa: E402
import tsem as TS  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402
from ensemble import rebalance  # noqa: E402

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
CORE = "car_ov17_fs2_t0.1_le_k2"


def pc_basis(Ztr, d):  # S/tsem_confirm/package_ref.pc_basis
    m = Ztr.mean(0)
    U, S, Vt = svd(Ztr - m, full_matrices=False, lapack_driver="gesdd")
    U, Vt = svd_flip(U, Vt, u_based_decision=False)
    return m, Vt[:d].T, S


def recorded(m):
    """Wrap the package's TSEM stage to record (view, input, output) per window (undo: del m.tsem.step)."""
    rec, step = [], type(m.tsem).step.__get__(m.tsem)
    m.tsem.step = lambda u, q: rec.append((u, q, step(u, q))) or rec[-1][2]
    return rec


def feed(m, X, bs=64):
    rec = recorded(m)
    m.reset_state()
    for i in range(0, len(X), bs):
        m.predict(X[i:i + bs])
    del m.tsem.step
    return np.array([o for _, _, o in rec])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src-harness", type=Path, required=True)
    ap.add_argument("--npc", type=int, default=128)
    for k, v in (("lam", 0.15), ("rho", 0.7), ("shrink", 0.7), ("m0", 20), ("n0", 60), ("nmin", 15)):
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--code", type=Path, default=SRC)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in ("lam", "rho", "shrink", "m0", "n0", "nmin")}
    with np.load(a.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    assert not any(k.startswith("tsem_") for k in z) and "em_lam" in z
    c = ts_core.parse(CORE)
    assert (int(z["ts_car"]), int(z["ts_nsub"]), float(z["ts_k"]), float(z["ts_shrink"])) == (1, 2, c["k"], 0.1)
    assert np.array_equal(z["ts_bands"], np.array(ts_core.BANDSETS[c["bands"]], dtype=np.float64))
    Hs = dict(np.load(a.src_harness))
    d = protocol.load()
    st, rows, N = d["stream"], np.ones(len(d["y"]), bool), len(d["y"])
    res = {"source": str(a.src), "source_weights_sha256": sha(a.src / "weights.npz"),
           "source_harness_sha256": sha(a.src_harness), "core": CORE, "npc": a.npc, "cfg": cfg,
           "code_sha256": sha(a.code), "ts_core_md5": hashlib.md5((ROOT / "experiments/ts_core.py").read_bytes()).hexdigest(),
           "tsem_md5": hashlib.md5((ROOT / "experiments/tsem.py").read_bytes()).hexdigest(),
           "git_rev": os.environ.get("NI26_SOURCE_REV"), "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID")}

    # the core's training features (ts_core.Feats, as cand_rec_ts's export) and its packaged scaler / priors
    t0 = time.time()
    fe = ts_core.Feats(d, c)
    sc = StandardScaler().fit(fe.F)
    priors = [fe.prior(b, np.arange(N)) for b in range(len(fe.C))]
    res.update({"feats_seconds": round(time.time() - t0, 1), "dim": int(fe.F.shape[1]),
                "scaler_mean_bit_identical": bool(np.array_equal(sc.mean_, z["ts_mu"])),
                "scaler_scale_bit_identical": bool(np.array_equal(sc.scale_, z["ts_sd"])),
                "scaler_max_rel_diff": float(max(np.abs(sc.mean_ - z["ts_mu"]).max() / np.abs(z["ts_mu"]).max(),
                                                 np.abs(sc.scale_ / z["ts_sd"] - 1).max())),
                "priors_bit_identical": bool(np.array_equal(np.stack(priors), z["ts_log_glob"]))})
    print(json.dumps(res, indent=1), flush=True)
    assert res["scaler_max_rel_diff"] < 1e-12 and res["priors_bit_identical"], res
    Ztr = (fe.F - z["ts_mu"]) / z["ts_sd"]  # = StandardScaler.transform with the packaged core's scaler
    t0 = time.time()
    m_tr, V, S = pc_basis(Ztr, a.npc)
    res.update({"svd_seconds": round(time.time() - t0, 1), "rank": int(len(S)), "S_head": S[:3].tolist(),
                "S_npc": float(S[a.npc - 1]), "pc_energy": float((S[:a.npc] ** 2).sum() / (S ** 2).sum())})
    del Ztr
    V32 = V.astype(np.float32)
    d2p = [d2.mean() for d2 in fe.d2]
    Zc = np.empty_like(fe.F)
    for s in dict.fromkeys(st):  # reference causal views (ts_core, prior = all windows, as the package)
        r = np.flatnonzero(st == s)
        Zc[r] = (fe.causal(r, priors, d2p) - z["ts_mu"]) / z["ts_sd"]
    del fe
    U64, U32 = (Zc - m_tr) @ V, (Zc - m_tr) @ V32.astype(np.float64)

    # package
    zn = {**z, **{f"tsem_{k}": np.array(v) for k, v in cfg.items()}, "tsem_m": m_tr, "tsem_V": V32}
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "weights.npz", **zn)
    for f in a.src.iterdir():
        if f.is_file() and f.name not in ("weights.npz", "submission.py"):
            shutil.copyfile(f, a.out / f.name)
    shutil.copyfile(a.code, a.out / "submission.py")

    # harness: SRC's mix -> rebalancing -> session EM (set_params.harness) -> batch TSEM (experiments/tsem.py)
    X64 = d["X"].astype(np.float64)
    Fe = np.c_[session_em.gaze_features(X64, list(d["ch_names"])),
               session_em.event_features(X64, list(d["ch_names"]), st, z["em_ev_g"])]
    Q = np.concatenate([rebalance(Hs["mix"][st == s], float(z["rb_alpha"]), float(z["rb_k"]), int(z["rb_w"]))
                        for s in dict.fromkeys(st)])
    kw = {k: float(z[f"em_{k}"]) for k in ("lam", "n0", "m0", "shrink")}
    E = session_em.apply(Fe, Q, st, iters=int(z["em_iters"]), nmin=int(z["em_nmin"]), **kw)
    assert np.array_equal(E.argmax(1), Hs["ref"]), "session EM differs from SRC's harness"
    ref32 = TS.tsem(E, st, TS.static(U32), **cfg)
    ref64 = TS.tsem(E, st, TS.static(U64), **cfg)

    m = load_solver(a.out, "new_pkg")
    rec = recorded(m)
    t0 = time.time()
    new = stream(m, d, rows, lambda mm, x: mm.predict(x))
    res["package_cpu_seconds_per_window"] = round((time.time() - t0) / N, 4)
    del m.tsem.step
    Up, Qp, Op = (np.array([r[i] for r in rec]) for i in range(3))
    old = Hs["new"]
    score = lambda lab: protocol.score(d["y"], lab, d, rows)["balanced_accuracy_cell_mean"]
    res.update({"n_windows": int(N), "view_max_abs_diff_vs_ref_f32_basis": float(np.abs(Up - U32).max()),
                "view_max_abs_diff_vs_ref_f64_basis": float(np.abs(Up - U64).max()),
                "em_output_max_abs_diff": float(np.abs(Qp - E).max()),
                "package_max_abs_prob_diff_vs_batch_tsem_f32": float(np.abs(Op - ref32).max()),
                "package_matches_harness": float(np.mean(new == ref32.argmax(1))),
                "n_windows_differ_from_harness": int(np.sum(new != ref32.argmax(1))),
                "package_max_abs_prob_diff_vs_batch_tsem_f64": float(np.abs(Op - ref64).max()),
                "label_flips_vs_f64_basis": int(np.sum(new != ref64.argmax(1))),
                "labels_changed_vs_source": int(np.sum(new != old)),
                "labels_changed_by_tsem_stage": int(np.sum(new != E.argmax(1))),
                "in_sample_cell_mean": {"source": score(old), "package": score(new)}})
    print(json.dumps(res, indent=1), flush=True)

    # causality and reset on two sessions, in batches of 64 as the evaluator feeds them
    s1, s2 = [np.flatnonzero(st == s) for s in list(dict.fromkeys(st))[:2]]
    T = 64 * ((len(s1) - 1) // 64)
    full = feed(m, d["X"][s1])
    trunc = feed(m, d["X"][s1[:T]])
    cp = copy.deepcopy(m)  # evaluator: deep copy at the session boundary, then reset_state
    res.update({"truncation_T": int(T), "truncated_bit_identical": bool(np.array_equal(full[:T], trunc)),
                "reset_deepcopy_bit_identical": bool(np.array_equal(feed(cp, d["X"][s2]),
                                                                    feed(load_solver(a.out, "fresh"), d["X"][s2]))),
                "batched_vs_streamed_max_abs_diff": float(np.abs(full - Op[s1]).max()),
                "files_sha256": {f.name: sha(f) for f in sorted(a.out.iterdir()) if f.is_file()}})
    np.savez(a.out.parent / f"{a.out.name}_harness.npz", E=E, ref32=ref32, ref64=ref64, new=new, old=old, Up=Up,
             U32=U32, Op=Op, S=S)
    (a.out.parent / f"{a.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["package_matches_harness"] == 1.0 and res["em_output_max_abs_diff"] < 1e-9
    assert res["truncated_bit_identical"] and res["reset_deepcopy_bit_identical"]


if __name__ == "__main__":
    main()
