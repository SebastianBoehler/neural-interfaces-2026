"""Timed B1 Solver replay on public EEG with no task labels."""
import ctypes
import importlib.util
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, "/official/tracks/bci_decoding")
from benchmark_utils.streaming import fresh_model

FIELDS = ("windows", "recognized", "ft_refits", "ft_steps", "ft_relabels",
          "ft_capped", "ft_over", "cp_refits", "cp_fused", "errors")


def read_source():
    with np.load("/input.npz", allow_pickle=False) as data:
        x = torch.from_numpy(data["X"])
        subjects, sessions = data["subject"], data["session"]
        channels = data["ch_names"].tolist()
    if x.shape != (2655, 41, 480) or x.dtype != torch.float32 or len(channels) != 41:
        raise ValueError("public worker source schema changed")
    streams = []
    for index, (person, session) in enumerate(sorted(set(zip(subjects, sessions)))):
        rows = np.flatnonzero((subjects == person) & (sessions == session))
        order = np.random.default_rng(np.random.SeedSequence([36017, index])).permutation(rows)
        streams.append((f"{person.rsplit('/', 1)[-1]}-{session}", order))
    if len(streams) != 14:
        raise ValueError("expected 14 complete public sessions")
    return x, streams, channels


def load(directory, channels):
    spec = importlib.util.spec_from_file_location("runtime_solver", directory / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solver().load_model({"submission_dir": directory, "device": "cuda:0",
                                      "ch_names": channels})


def activity(model):
    if hasattr(model, "handle"):
        observe = ctypes.CDLL("/work/activity.so").ni26_observe
        observe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
        observe.restype = None
        output = (ctypes.c_int64 * len(FIELDS))()
        observe(model.handle, output)
        return dict(zip(FIELDS, output))
    log, ft, cp, cf = model.log6, model.ft, model.cp, model.cf
    return dict(zip(FIELDS, (log["n"], log["rec"] if log["rec"] is not None else -1,
                            ft.refits, ft.steps, ft.relabels, ft.capped, ft.over, cp.refits,
                            cp.fused, log["err"]+cp.log["err"]+cf.log["err"])))


def stream(template, x, rows):
    torch.cuda.synchronize()
    start = perf_counter()
    model = fresh_model(template)
    created = perf_counter()
    labels = []
    for row in rows:
        window = x[int(row):int(row)+1].to("cuda:0")
        label = model.predict(window)
        if label.device.type != "cpu" or label.dtype != torch.int64 or label.shape != (1,):
            raise ValueError("expected one CPU int64 label per call")
        labels.append(int(label[0]))
    torch.cuda.synchronize()
    info = activity(model)
    del model
    torch.cuda.synchronize()
    finish = perf_counter()
    if info["errors"] or info["windows"] != len(rows) or min(labels) < 0 or max(labels) > 2:
        raise ValueError(f"worker stream failed: {info}")
    return {"s": finish-start, "copy_s": created-start, "labels": labels, "activity": info}


def main(suite, mode, output):
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("expected one A100")
    wall = perf_counter()
    x, streams, channels = read_source()
    if mode == "normal":
        if suite == "smoke":
            streams = [("P01-canonical31", np.sort(streams[0][1])[:31])]
    elif mode == "stress":
        rows = np.random.default_rng(36017800).permutation(len(x))
        lengths = (64,) if suite == "smoke" else (378, 800)
        streams = [(f"forced-unknown-{n}", rows[:n]) for n in lengths]
    else:
        raise ValueError("unknown worker mode")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    model = load(Path("/package"), channels)
    torch.cuda.synchronize()
    load_seconds = perf_counter()-start
    result = {"suite": suite, "mode": mode, "load_s": load_seconds, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "streams": {}}
    rounds = ("first", "repeat") if suite == "smoke" else ("first",)
    for repeat in rounds:
        entries = {}
        for name, rows in streams:
            entries[name] = row = stream(model, x, rows)
            info = row["activity"]
            if mode == "stress" and (info["recognized"] != -1 or info["ft_refits"] <= 0
                    or suite == "full" and (info["cp_fused"] <= 0 or info["ft_capped"] <= 0)):
                raise RuntimeError(f"expected active unrecognized adaptation: {name} {info}")
            print(repeat, name, round(row["s"], 5), info, flush=True)
        result["streams"][repeat] = entries
        Path(output).write_text(json.dumps(result, indent=2)+"\n")
    result.update(wall_s=perf_counter()-wall, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  cuda_live_bytes=torch.cuda.max_memory_allocated(),
                  cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    if "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text():
        raise RuntimeError("worker mapped a host environment dependency")
    Path(output).write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
