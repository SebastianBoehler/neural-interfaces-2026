"""Lane D3 NETBCI stress: enrol day 1 of person k (moments d3 included) into hardcbB ONCE, then stream days 2-4 through the real Solver of every arm
(D3_ARMS, default hardcbB+D3f+D3g; D0z = code in, mode 0) on identical tapes, one window per call, fresh model per (arm, order, day).
usage: run.py BASE.zip EXT_DIR K OUT.npz ; env B10_OFF, B10_ORDERS (e.g. full:72000,full:72001), B10_DAYS (default 2,3,4), D3_ARMS.
Same enrolment recipe as laneB10/run.py (MI -> 0, rest alternating 1/2). Day 2-4 labels are never opened by the stream; y is stored for scoring only.
Fails loudly: D0z must equal hardcbB label for label; a D3 arm may differ from hardcbB only AFTER the recognition window (and only if the
drift stage was active); a stream that recognised and routed windows must report them."""
import os
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["B10_OFF"])
sys.path.insert(0, str(HERE.parent / "niv"))
sys.path.insert(0, str(HERE))
from benchmark_utils.streaming import fresh_model  # noqa: E402
import build_drift  # noqa: E402
import enrol  # noqa: E402
import orders  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("b10_run", HERE.parent / "laneB10" / "run.py")
_b10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b10)  # same package loader and recognition read-out as lane B10
load, recognised = _b10.load, _b10.recognised


def arm_dir(root, base, arm):
    """package directory of an arm: symlinks to the enrolled package files, submission.py patched (hardcbB: the enrolled package itself)."""
    if arm == "hardcbB":
        return base
    d = Path(root) / arm
    d.mkdir()
    for f in base.iterdir():
        if f.name == "submission.py":
            (d / f.name).write_bytes(build_drift.patch(f.read_bytes(), arm))
        else:
            (d / f.name).symlink_to(f)
    return d


def main(base, ext, k, out):
    assert torch.cuda.is_available()
    arms = os.environ.get("D3_ARMS", "hardcbB+D3f+D3g").split("+")
    assert arms[0] == "hardcbB"
    with np.load(Path(ext) / f"p{k}.npz", allow_pickle=False) as z:
        X, y, day, run, onset, name = z["X"], z["y"], z["day"], z["run"], z["onset"], str(z["name"])
        channels = z["ch_names"].tolist()
    assert len(channels) == 41
    with TemporaryDirectory() as tmp:
        rows = np.flatnonzero(day == 1)
        rows = rows[np.lexsort((onset[rows], run[rows]))]
        ye = np.where(y[rows] == 1, 0, 1 + (np.cumsum(y[rows] == 0) % 2))
        assert set(ye) == {0, 1, 2}
        zin = f"{tmp}/enrolled.zip"
        print("enrolled", name, enrol.enrol(base, zin, X[rows], ye, "N" + str(k), d3=True), "people", len(rows), flush=True)
        with zipfile.ZipFile(zin) as zf:
            zf.extractall(f"{tmp}/pkg")
        with np.load(f"{tmp}/pkg/weights.npz") as w:
            assert np.isfinite(w["d3_mu"][-1]).all() and np.isnan(w["d3_mu"][:-1]).all(), "d3 rows: enrolled finite, public NaN"
            print("d3 row", np.round(w["d3_n"][-1]), "var mean", float(w["d3_var"][-1].mean()), flush=True)
        templates = {a: load(arm_dir(f"{tmp}", Path(tmp) / "pkg", a), channels) for a in arms}
        xt = torch.from_numpy(X)
        stream = np.array([f"{name}|{d}" for d in day])
        result = {}
        for o in os.environ["B10_ORDERS"].split(","):
            scheme, s = o.split(":")
            p = orders.permutation(stream, run, scheme, int(s))
            plan = {v: p[stream[p] == v] for v in sorted(set(stream))}
            for d in os.environ.get("B10_DAYS", "2,3,4").split(","):
                rws, lab = plan[f"{name}|{d}"], {}
                for a in arms:
                    t0, model, labels = time.perf_counter(), fresh_model(templates[a]), []
                    for r in rws:
                        l = model.predict(xt[int(r):int(r) + 1].to("cuda:0"))
                        if l.dtype != torch.int64 or l.shape != (1,):
                            raise ValueError("expected one int64 label per call")
                        labels.append(int(l[0]))
                    rec, routed, err = recognised(model)
                    if err or not 0 <= min(labels) <= max(labels) <= 2:
                        raise RuntimeError(f"stream failed {a} {o} day {d}: err {err}")
                    lab[a] = np.array(labels, np.int8)
                    st = getattr(model, "d3", None)
                    st = dict(st.stats) if st is not None else {}
                    sec = time.perf_counter() - t0
                    tag = f"{a}|{scheme}|{s}|{d}"
                    result[tag + "|labels"] = lab[a]
                    result[tag + "|meta"] = np.array([rec, routed, err, round(sec, 2), st.get("routed", -1), st.get("active", -1), st.get("short", -1),
                                                      st.get("uncorroborated", -1), st.get("nan_row", -1), st.get("r_sum", 0.0), st.get("max_corr", 0.0)], np.float64)
                    diff = np.flatnonzero(lab[a] != lab["hardcbB"])
                    if a == "D0z" and len(diff):
                        raise RuntimeError(f"D0z differs from hardcbB at {diff[:5]}: the switch is not a zero change")
                    if a in ("D3f", "D3g") and len(diff) and (rec < 0 or diff[0] <= rec or st["active"] <= 0):
                        raise RuntimeError(f"{a} differs from hardcbB at {diff[0]} (recognised at {rec}, active {st.get('active')}): not post-latch")
                    print("PASS", a, o, d, "rec", rec, "routed", routed, "ndiff", len(diff), "drift", st, "s", round(sec, 1), flush=True)
                    del model
                result[f"{scheme}|{s}|{d}|rows"] = rws.astype(np.int64)
                result[f"{scheme}|{s}|{d}|y"] = y[rws]
    np.savez(out, **result)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
