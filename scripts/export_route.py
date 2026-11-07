"""Build cand_rec_ts2_tsem_route3geb (7 Nov): cand_rec_ts2_tsem plus returning-person routing v3 and the gated early boost.

Usage: python scripts/export_route.py SRC_DIR OUT_DIR --route route_pkg3.npz --hi HI_DIR --src-harness H --hi-harness H2
       [--code submissions/ensemble/submission.py] [--off-only] [--set KEY=VALUE ...]
--set changes or adds non-route keys of SRC's weights before the build; route3geb uses em_nmin=20 em_boost=2.0
tsem_boost=4.0 eb_T=30.0 eb_gate_lo=0.20 eb_gate_hi=0.35. Reference: experiments/route_stream3geb.py (V3TailGEB; without
the boost / gate keys it is the plain v3 reference bitwise). Routing v1 / v2 and their references: commit c4d600d.
SRC_DIR: cand_rec_ts2_tsem (TSEM PC128, lam 0.15: the stage's input). HI_DIR: cand_rec_ts2_tsem256; its tsem_m and the
first 128 columns of its tsem_V must equal SRC's; route_V_hi = its tsem_V[:, 128:] (float32), so the stage's view is
the PC256 view [TSEM view, (z - tsem_m) @ route_V_hi]. route_pkg3.npz: the route_* keys in that basis (route_v3 = 1:
gate on the unrouted (shadow) tail, routed rebalancing -> SessionEM -> TSEM tail of its own; its other keys,
rb_* / em_* / tsem_*, must equal SRC's and are not written). H / H2: SRC's and HI's export harnesses.
Package: SRC + route_* keys + route_V_hi (+ --set); every other key and file unchanged; submission.py is --code.
Checks on all public windows, streamed per session: the shadow input vs SRC's harness (TSEM output ref32, labels); its
views vs HI's harness views; its routed and shadow outputs vs the reference on the stage's own inputs; the GEB wiring
(each tail's TSEM reads its own tail's SessionEM gate); every public session routes to its own person from route_t_start
on; on the first two sessions in batches of 64, truncated streams are bit-identical before T and a deep-copied, reset
model equals a fresh one. --off-only: a temporary copy with route_tau 1e9 must give labels and final posteriors
byte-identical to SRC (+ --set, without route keys, --code; or SRC with its own code when no --set). Writes
OUT_DIR_check.json and OUT_DIR_harness.npz (or OUT_DIR_off.json).
"""
import argparse
import copy
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
import route_stream3geb as RS  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402

sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
KEYS3 = ("route_gal", "route_pop", "route_b", "route_Q", "route_names", "route_clip", "route_rho", "route_t_start",
         "route_tau", "route_ps1_W", "route_ps1_c", "route_ramp", "route_slot_g", "route_slot_a", "route_v3")


def recorded(m, stage):
    """Wrap m.<stage>.step to record (args, output, last, SessionEM features of the stage's own tail) per window."""
    st = getattr(m, stage)
    rec, step = [], type(st).step.__get__(st)

    def f(*args):
        o = step(*args)
        em = getattr(st, "em", None)
        rec.append((args, o, getattr(st, "last", None), None if em is None else np.array(em.A[-1]),
                    em.gates[-1] if hasattr(em, "gates") else None))
        return o
    st.step = f
    return rec


def feed(m, X, bs=64):
    rec = recorded(m, "route")
    m.reset_state()
    for i in range(0, len(X), bs):
        m.predict(X[i:i + bs])
    del m.route.step
    return np.array([r[1] for r in rec])


def build(src, out, z, extra, code):
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "weights.npz", **z, **extra)
    for f in src.iterdir():
        if f.is_file() and f.name not in ("weights.npz", "submission.py"):
            shutil.copyfile(f, out / f.name)
    shutil.copyfile(code, out / "submission.py")


def run(folder, tag, stage, d, rows):
    m = load_solver(folder, tag)
    rec = recorded(m, stage)
    t0 = time.time()
    lab = stream(m, d, rows, lambda mm, x: mm.predict(x))
    secs = (time.time() - t0) / rows.sum()
    delattr(getattr(m, stage), "step")
    return m, lab, rec, secs


def reference(zn, st, Q, MIX, C1, F, U):
    """Reference routed and shadow outputs per window, streamed per session."""
    v = RS.V3TailGEB(zn)
    O, S = np.empty_like(Q), np.empty_like(Q)
    for s in dict.fromkeys(st):
        v.reset_state()
        for i in np.flatnonzero(st == s):
            O[i], S[i] = v.step(MIX[i], C1[i], F[i], U[i])
    return O, S


def main():
    ap = argparse.ArgumentParser()
    for k in ("src", "out"):
        ap.add_argument(k, type=Path)
    for k in ("route", "hi", "src-harness", "hi-harness"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--code", type=Path, default=SRC)
    ap.add_argument("--off-only", dest="off_only", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="change / add a non-route key of SRC's weights (e.g. the early boost: em_nmin=20 em_boost=2)")
    a = ap.parse_args()
    with np.load(a.src / "weights.npz") as f:
        z = {k: f[k] for k in f.files}
    sets = dict(s.split("=", 1) for s in a.set)
    assert not any(k.startswith("route_") for k in sets)
    z_src, z = z, {**z, **{k: np.array(v, dtype=z[k].dtype if k in z else np.float64) for k, v in sets.items()}}
    with np.load(a.hi / "weights.npz") as f:
        m_hi, V_hi = f["tsem_m"], f["tsem_V"]
    with np.load(a.route) as f:
        rp = {k: f[k] for k in f.files}
    assert int(rp.get("route_v3", 0)) == 1, "routing v1 / v2 packages: use commit c4d600d"
    other = {k: rp.pop(k) for k in [k for k in rp if not k.startswith("route_")]}  # copies of SRC's tail scalars
    assert all(np.array_equal(z_src[k], v) for k, v in other.items())
    assert sorted(rp) == sorted(KEYS3) and "tsem_lam" in z
    assert not any(k.startswith("route_") for k in z)
    d128 = z["tsem_V"].shape[1]
    assert np.array_equal(m_hi, z["tsem_m"]) and V_hi[:, :d128].tobytes() == z["tsem_V"].tobytes()
    extra = {**rp, "route_V_hi": np.ascontiguousarray(V_hi[:, d128:])}
    assert d128 + extra["route_V_hi"].shape[1] == rp["route_gal"].shape[2]
    zn = {**z, **extra}
    d = protocol.load()
    st, rows, N = d["stream"], np.ones(len(d["y"]), bool), len(d["y"])
    names = [str(n) for n in rp["route_names"]]
    res = {"source": str(a.src), "source_weights_sha256": sha(a.src / "weights.npz"), "hi": str(a.hi),
           "hi_weights_sha256": sha(a.hi / "weights.npz"), "route_npz_sha256": sha(a.route), "route_version": 3,
           "code_sha256": sha(a.code), "reference": RS.__file__,
           "reference_md5": hashlib.md5(Path(RS.__file__).read_bytes()).hexdigest(),
           "route_npz_other_keys_equal_src": sorted(other), "names": names, "set": sets,
           "scalars": {k: float(v) for k, v in rp.items() if v.ndim == 0},
           "git_rev": os.environ.get("NI26_SOURCE_REV"), "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    Hs, Hh = np.load(a.src_harness), np.load(a.hi_harness)

    if a.off_only:  # routing off vs the unrouted package: labels and final posteriors byte-identical
        off = {"route_tau": np.array(1e9)}
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as tmp2:
            build(a.src, Path(tmp), z, {**extra, **off}, a.code)
            _, lab0, rec0, _ = run(Path(tmp), "off", "route", d, rows)
            if sets:  # target: SRC with the same --set changes, no route keys, --code
                build(a.src, Path(tmp2), z, {}, a.code)
                _, labs, recs, _ = run(Path(tmp2), "src_set", "tsem", d, rows)
        if not sets:  # target: SRC with its own code
            _, labs, recs, _ = run(a.src, "src", "tsem", d, rows)
        O0, Os = np.array([r[1] for r in rec0]), np.array([r[1] for r in recs])
        res.update({"off": {k: float(v) for k, v in off.items()},
                    "off_target": "SRC + --set, no route keys, --code" if sets else "SRC with its own code",
                    "off_labels_byte_identical_to_src": bool(np.array_equal(lab0, labs)),
                    "off_final_posteriors_byte_identical_to_src": bool(np.array_equal(O0, Os)),
                    "off_stage_output_is_input_bitwise": bool(all(np.array_equal(r[0][-1], r[1]) for r in rec0)),
                    "off_never_routed": bool(all(r[2] is None or r[2][2] == 0 for r in rec0)),
                    "src_labels_equal_src_harness": bool(np.array_equal(labs, Hs["new"]))})
        (a.out.parent / f"{a.out.name}_off.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        assert res["off_labels_byte_identical_to_src"] and res["off_final_posteriors_byte_identical_to_src"]
        return

    build(a.src, a.out, z, extra, a.code)
    m, new, rec, spw = run(a.out, "new_pkg", "route", d, rows)
    res["slot_a_equals_core1_share"] = bool(m.w[0] == float(rp["route_slot_a"]))
    att = lambda o: {k: getattr(o, k, None) for k in ("nmin", "lam", "boost", "boost_T", "g_lo", "g_hi")}
    res["routed_tail_stages_equal_shadow"] = {"em": att(m.route.em), "tsem": att(m.route.tsem),
                                              "equal": bool(att(m.route.em) == att(m.em) and att(m.route.tsem) == att(m.tsem))}
    res["gate_wiring"] = {"shadow_tsem_reads_shadow_em": m.tsem.gate_src is m.em,  # GEB: each tail's own SessionEM
                          "routed_tsem_reads_routed_em": m.route.tsem.gate_src is m.route.em,
                          "routed_em_is_not_shadow_em": m.route.em is not m.em}
    assert all(res["gate_wiring"].values()), res["gate_wiring"]
    assert res["slot_a_equals_core1_share"] and res["routed_tail_stages_equal_shadow"]["equal"]
    arg = lambda i: np.array([r[0][i] for r in rec])
    O, last = np.array([r[1] for r in rec]), [r[2] for r in rec]
    U, MIX, C1, Q, F = arg(1), arg(2), arg(3), arg(4), np.array([r[3] for r in rec])
    ref, ref_sh = reference(zn, st, Q, MIX, C1, F, U)
    score = lambda lab: protocol.score(d["y"], lab, d, rows)["balanced_accuracy_cell_mean"]
    t0r, routing = int(rp["route_t_start"]), {}
    for s in dict.fromkeys(st):
        r = np.flatnonzero(st == s)
        own = s.split("|")[0].split("/")[-1]
        L = [last[i] for i in r]
        on = [k for k, x in enumerate(L) if x is not None and x[2] > 0]
        routing[s] = {"n": int(len(r)), "persons": sorted({names[x[0]] for x in L if x is not None}),
                      "first_routed": on[0] if on else None, "min_score": float(min(x[1] for x in L if x is not None)),
                      "mean_weight_routed": float(np.mean([L[k][2] for k in on])) if on else 0.0,
                      "own_from_t_start": bool(all(x is None for x in L[:t0r]) and len(on) == len(r) - t0r
                                               and all(names[L[k][0]] == own for k in on) and on[:1] == [t0r])}
    res.update({"n_windows": int(N), "package_cpu_seconds_per_window": round(spw, 4),
                "stage_input_max_abs_diff_vs_src_harness_ref32": float(np.abs(Q - Hs["ref32"]).max()),
                "stage_input_labels_equal_src_labels": bool(np.array_equal(Q.argmax(1), Hs["new"])),
                "view_max_abs_diff_vs_hi_harness_U32": float(np.abs(U - Hh["U32"]).max()),
                "view_first128_max_abs_diff_vs_src_harness_U32": float(np.abs(U[:, :d128] - Hs["U32"]).max()),
                "package_max_abs_diff_vs_reference": float(np.abs(O - ref).max()),
                "package_bit_identical_to_reference": bool(np.array_equal(O, ref)),
                "label_flips_vs_reference": int((O.argmax(1) != ref.argmax(1)).sum()),
                "labels_eq_output_argmax": bool(np.array_equal(new, O.argmax(1))),
                "labels_changed_vs_source": int((new != Hs["new"]).sum()),
                "labels_changed_by_route_stage": int((new != Q.argmax(1)).sum()),
                "all_sessions_own_person_from_t_start": bool(all(v["own_from_t_start"] for v in routing.values())),
                "routing": routing, "in_sample_cell_mean": {"source": score(Hs["new"]), "package": score(new)}})
    res.update({"shadow_max_abs_diff_vs_reference": float(np.abs(Q - ref_sh).max()),
                "shadow_bit_identical_to_reference": bool(np.array_equal(Q, ref_sh)),
                "shadow_label_flips_vs_reference": int((Q.argmax(1) != ref_sh.argmax(1)).sum())})
    gt = np.array([np.nan if r[4] is None else r[4] for r in rec])  # routed tail's GEB gate per window
    pos = np.empty(N, int)
    for s in dict.fromkeys(st):
        pos[st == s] = np.arange((st == s).sum())
    res["routed_gate_mean_t15_59"] = float(gt[(pos >= 15) & (pos < 60)].mean())
    res["routed_gate_max_t15_59"] = float(gt[(pos >= 15) & (pos < 60)].max())
    print(json.dumps(res, indent=1), flush=True)

    s1, s2 = [np.flatnonzero(st == s) for s in list(dict.fromkeys(st))[:2]]
    T = 64 * ((len(s1) - 1) // 64)
    full = feed(m, d["X"][s1])
    trunc = feed(m, d["X"][s1[:T]])
    cp = copy.deepcopy(m)  # evaluator: deep copy at the session boundary, then reset_state
    res.update({"truncation_T": int(T), "truncated_bit_identical": bool(np.array_equal(full[:T], trunc)),
                "reset_deepcopy_bit_identical": bool(np.array_equal(feed(cp, d["X"][s2]),
                                                                    feed(load_solver(a.out, "fresh"), d["X"][s2]))),
                "batched_vs_streamed_max_abs_diff": float(np.abs(full - O[s1]).max()),
                "files_sha256": {f.name: sha(f) for f in sorted(a.out.iterdir()) if f.is_file()}})
    np.savez(a.out.parent / f"{a.out.name}_harness.npz", U=U, Q=Q, O=O, ref=ref, new=new, old=Hs["new"],
             MIX=MIX, C1=C1, F=F, ref_shadow=ref_sh,
             last=np.array([x if x is not None else (-1, np.nan, np.nan) for x in last], dtype=np.float64))
    (a.out.parent / f"{a.out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "routing"}, indent=1))
    assert res["package_max_abs_diff_vs_reference"] <= 1e-9 and res["label_flips_vs_reference"] == 0
    assert res["shadow_max_abs_diff_vs_reference"] <= 1e-9 and res["shadow_label_flips_vs_reference"] == 0
    assert res["all_sessions_own_person_from_t_start"] and res["labels_eq_output_argmax"]
    assert res["truncated_bit_identical"] and res["reset_deepcopy_bit_identical"]


if __name__ == "__main__":
    main()
