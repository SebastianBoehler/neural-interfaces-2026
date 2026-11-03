"""Copy a sealed package with new rebalancing / session-EM parameters (data-only arm, 3 Nov).

Usage: python scripts/set_params.py SRC_DIR OUT_DIR rb_alpha=1.15 em_shrink=0.85 em_n0=40 ...
Keys: rb_alpha rb_k rb_w em_lam em_n0 em_m0 em_shrink em_iters em_nmin. Every other file and array is
copied unchanged (submission.py stays the frozen code). Checks every public window, streamed per
session, as export_external_teacher.py: the package's member mix (teacher included) -> rebalance ->
session_em.apply equals Solver.predict, for the source parameters on SRC and the new ones on OUT.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
import session_em  # noqa: E402
from add_rebalance import load_solver, stream  # noqa: E402
from ensemble import rebalance  # noqa: E402

KEYS = {"rb_alpha": np.float64, "rb_k": np.float64, "rb_w": np.int64, **{
    f"em_{k}": np.float64 for k in ("lam", "n0", "m0", "shrink", "iters", "nmin")}}


def harness(z, mix, st, Fe):
    Q = np.concatenate([rebalance(mix[st == s], float(z["rb_alpha"]), float(z["rb_k"]), int(z["rb_w"]))
                        for s in dict.fromkeys(st)])
    kw = {k: float(z[f"em_{k}"]) for k in ("lam", "n0", "m0", "shrink")}
    return session_em.apply(Fe, Q, st, iters=int(z["em_iters"]), nmin=int(z["em_nmin"]), **kw).argmax(1)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    new = {k: KEYS[k](v) for k, v in (a.split("=") for a in sys.argv[3:])}
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "weights.npz":
            shutil.copyfile(f, out / f.name)
    z = dict(np.load(src / "weights.npz"))
    old = {k: z[k].item() for k in new}
    np.savez(out / "weights.npz", **{**z, **new})

    d = protocol.load()
    st, rows = d["stream"], np.ones(len(d["y"]), bool)
    ch, X64 = list(d["ch_names"]), d["X"].astype(np.float64)
    Fe = np.c_[session_em.gaze_features(X64, ch), session_em.event_features(X64, ch, st, z["em_ev_g"])]
    mix = stream(load_solver(src, "src_mix"), d, rows, lambda m, x: m.mixed_proba(x))
    p_src = stream(load_solver(src, "src_pkg"), d, rows, lambda m, x: m.predict(x))
    p_new = stream(load_solver(out, "new_pkg"), d, rows, lambda m, x: m.predict(x))
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    res = {"source": str(src), "changed": {k: [old[k], new[k].item()] for k in new},
           "source_matches_harness": float(np.mean(p_src == harness(z, mix, st, Fe))),
           "package_matches_harness": float(np.mean(p_new == harness({**z, **new}, mix, st, Fe))),
           "labels_changed_vs_source": int(np.sum(p_new != p_src)), "n_windows": int(len(p_new)),
           "files_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()}}
    (out.parent / f"{out.name}_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["source_matches_harness"] == 1.0 and res["package_matches_harness"] == 1.0


if __name__ == "__main__":
    main()
