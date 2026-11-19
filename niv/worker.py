"""NI-V worker: stream the visits of ONE Scherer person through an exact package ZIP, one window at a time.

python worker.py ZIP OUT.npz PERSON_INDEX ARMS
ARMS = comma list of scheme:seedindex (e.g. full:0,within:3). Every (arm, visit) runs on a fresh copy of the loaded
Solver model (official fresh_model), so nothing crosses visits. Labels only; no task label is read here.
Runs inside runs/cloud/ni26_worker.sif with /official, /external.npz, /rows.tsv.gz and /package bound.
"""
import importlib.util
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, "/official/tracks/bci_decoding")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_utils.streaming import fresh_model  # noqa: E402
import orders  # noqa: E402


def load(directory, channels):
    spec = importlib.util.spec_from_file_location("niv_solver", directory / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solver().load_model({"submission_dir": directory, "device": "cuda:0", "ch_names": channels})


def recognised(model):
    """Window index of first recognition (-1: never) for v5-style and v6-style models; errors count too."""
    log6 = getattr(model, "log6", None)
    err = 0
    if log6 is not None:
        err = int(log6.get("err", 0))
        rec = int(log6.get("rec", -1))
    else:
        rec = -1
    route = getattr(model, "route", None)
    routed = int(route.log["routed"]) if route is not None and hasattr(route, "log") else 0
    return rec, routed, err


def main(zip_path, out, person, arms):
    assert torch.cuda.is_available()
    with np.load("/external.npz", allow_pickle=False) as z:
        x = torch.from_numpy(z["X"])
        subject, session, channels = z["subject"], z["session"], z["ch_names"].tolist()
    run = orders.runs_from_manifest("/rows.tsv.gz", subject, session)
    who = sorted(set(subject))[int(person)]
    visits = [f"{who}|1", f"{who}|2"]
    result = {}
    with TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        template = load(Path(tmp), channels)
        for arm in arms.split(","):
            scheme, k = arm.split(":")
            seed = orders.SEEDS[int(k)]
            plan = orders.plans(subject, session, run, scheme, seed)
            for v in visits:
                rows = plan[v]
                t0 = time.perf_counter()
                model = fresh_model(template)
                labels = []
                for r in rows:
                    y = model.predict(x[int(r):int(r) + 1].to("cuda:0"))
                    if y.dtype != torch.int64 or y.shape != (1,):
                        raise ValueError("expected one int64 label per call")
                    labels.append(int(y[0]))
                rec, routed, err = recognised(model)
                if err or len(labels) != len(rows) or not 0 <= min(labels) <= max(labels) <= 2:
                    raise RuntimeError(f"stream failed {arm} {v}: err {err}")
                tag = f"{scheme}|{k}|{v}"
                result[tag + "|rows"] = rows.astype(np.int64)
                result[tag + "|labels"] = np.array(labels, np.int8)
                result[tag + "|meta"] = np.array([rec, routed, err, round(time.perf_counter() - t0, 2)], np.float64)
                print("PASS", tag, "rec", rec, "routed", routed, "s", round(time.perf_counter() - t0, 1), flush=True)
                del model
    np.savez(out, **result)


if __name__ == "__main__":
    main(*sys.argv[1:])
