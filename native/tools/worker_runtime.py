"""Run the real Solver entry point in one isolated worker process."""
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import torch

from runtime_reference import load_solver


def main():
    official, package, fixture, output = map(Path, sys.argv[1:])
    sys.path.insert(0, str(official))
    from benchmark_utils.streaming import fresh_model

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("expected the worker replica on an A100")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    model = load_solver(package, "cuda:0")
    torch.cuda.synchronize()
    load_seconds = perf_counter() - started
    x = torch.load(fixture, map_location="cpu", weights_only=True)["x"].cuda()
    if x.shape != (31, 41, 480):
        raise ValueError("expected the pinned P01 source prefix")
    elapsed, outputs = [], []
    for _ in range(2):
        torch.cuda.synchronize()
        started = perf_counter()
        stream = fresh_model(model)
        out = torch.cat([stream.predict(window[None]).cpu() for window in x])
        torch.cuda.synchronize()
        elapsed.append(perf_counter() - started)
        if out.shape != (len(x),) or out.dtype != torch.int64:
            raise RuntimeError("solver emitted an invalid label vector")
        outputs.append(out.tolist())
        del stream
    if outputs[0] != outputs[1]:
        raise RuntimeError("fresh session copies reproduced different labels")
    output.write_text(json.dumps({
        "labels": outputs[0], "cold_load_seconds": load_seconds, "prefix_seconds": elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_cuda_live_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
