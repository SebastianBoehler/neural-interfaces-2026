"""B10 NETBCI cross-day stress: enrol day 1 of NETBCI person k into a package, stream days 2-4 through the real Solver (one window per call).
LANE R4b copy of laneB10/run.py: env R4B_PAIRS "e>d,d,.." (enrol day > stream days; default NETBCI 1>2,3,4; ds007221g "1>2;2>1"), and the per-window proxy log of the gate packages is saved.
usage: run.py BASE.zip EXT_DIR K OUT.npz [--new] ; env B10_OFF (official bci_decoding dir), B10_ORDERS (e.g. full:72000,within:72000), B10_DAYS (default 2,3,4).
Labels of day 1 (enrolment only): right_hand -> task 0 (MI); rest -> tasks 1/2 by alternating rank (NETBCI has no task-1/2 classes; the 3-class
package code needs all three: placeholder split, no information). Days 2-4 labels are never opened here. --new: no enrolment (same stream through the base package)."""
import importlib.util
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
from benchmark_utils.streaming import fresh_model  # noqa: E402
import enrol  # noqa: E402
import orders  # noqa: E402


def load(directory, channels):
    spec = importlib.util.spec_from_file_location("b10_solver", directory / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solver().load_model({"submission_dir": directory, "device": "cuda:0", "ch_names": channels})


def recognised(model):
    log6 = getattr(model, "log6", None)
    err, rec = (int(log6.get("err", 0)), int(log6.get("rec", -1))) if log6 is not None else (0, -1)
    route = getattr(model, "route", None)
    routed = int(route.log["routed"]) if route is not None and hasattr(route, "log") else 0
    return rec, routed, err


def main(base, ext, k, out, new=False):
    pairs = [(int(a), [int(d) for d in b.split(",")]) for a, b in (g.split(">") for g in os.environ.get("R4B_PAIRS", "1>2,3,4").split(";"))]
    assert torch.cuda.is_available()
    with np.load(Path(ext) / f"p{k}.npz", allow_pickle=False) as z:
        X, y, day, run, onset, name = z["X"], z["y"], z["day"], z["run"], z["onset"], str(z["name"])
        channels = z["ch_names"].tolist()
    assert len(channels) == 41
    result = {}
    for eday, sdays in pairs:
      with TemporaryDirectory() as tmp:
        zin = base
        if not new:
            rows = np.flatnonzero(day == eday)
            rows = rows[np.lexsort((onset[rows], run[rows]))]
            ye = np.where(y[rows] == 1, 0, 1 + (np.cumsum(y[rows] == 0) % 2))  # MI 0; rest alternating 1/2
            assert set(ye) == {0, 1, 2}
            zin = f"{tmp}/enrolled.zip"
            print("enrolled", name, eday, enrol.enrol(base, zin, X[rows], ye, "N" + str(k)), "people", len(rows), "windows", flush=True)
        with zipfile.ZipFile(zin) as zf:
            zf.extractall(f"{tmp}/pkg")
        template = load(Path(tmp) / "pkg", channels)
        xt = torch.from_numpy(X)
        stream = np.array([f"{name}|{d}" for d in day])
        for arm in os.environ["B10_ORDERS"].split(","):
            scheme, s_ = arm.split(":")
            seed = int(s_)
            p = orders.permutation(stream, run, scheme, seed)
            plan = {v: p[stream[p] == v] for v in sorted(set(stream))}
            for d in sdays:
                assert d != eday
                rows, t0 = plan[f"{name}|{d}"], time.perf_counter()
                model, labels = fresh_model(template), []
                for r in rows:
                    lab = model.predict(xt[int(r):int(r) + 1].to("cuda:0"))
                    if lab.dtype != torch.int64 or lab.shape != (1,):
                        raise ValueError("expected one int64 label per call")
                    labels.append(int(lab[0]))
                rec, routed, err = recognised(model)
                if err or not 0 <= min(labels) <= max(labels) <= 2:
                    raise RuntimeError(f"stream failed {arm} day {d}: err {err}")
                tag = f"{scheme}|{seed}|{eday}{d}"
                result[tag + "|rows"] = rows.astype(np.int64)
                result[tag + "|y"] = y[rows]  # scoring only; never read by the stream above
                result[tag + "|labels"] = np.array(labels, np.int8)
                result[tag + "|meta"] = np.array([rec, routed, err, round(time.perf_counter() - t0, 2)], np.float64)
                px = getattr(model, "pxy_log", None)
                if px is not None:
                    assert len(px) == len(rows), (len(px), len(rows))
                    result[tag + "|pxy"] = np.array(px, np.float32)
                print("PASS", tag, "rec", rec, "routed", routed, "s", round(time.perf_counter() - t0, 1), flush=True)
                del model
    np.savez(out, **result)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], "--new" in sys.argv[5:])
