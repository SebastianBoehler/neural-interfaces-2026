"""Capture final scores from the frozen solver on unlabeled source windows."""
import importlib.util
from pathlib import Path
import sys

import torch


def load_solver(package, device):
    spec = importlib.util.spec_from_file_location("runtime_solver", package / "submission.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference.Solver().load_model({"submission_dir": package, "device": device})


def main():
    official, package, component_fixture, output = map(Path, sys.argv[1:])
    sys.path.insert(0, str(official))
    from benchmark_utils.streaming import fresh_model

    model = fresh_model(load_solver(package, "cpu"))
    x = torch.load(component_fixture, map_location="cpu", weights_only=True)["unknown_x"]
    if x.shape != (31, 41, 480) or x.dtype != torch.float32:
        raise ValueError("expected archived 31-window P01 prefix")
    scores = []
    step = model.cp.step

    def capture(*args):
        result = step(*args)
        scores.append(torch.as_tensor(result.copy()))
        return result

    model.cp.step = capture
    labels = torch.cat([model.predict(window[None]) for window in x])
    final = torch.stack(scores)
    if not torch.equal(labels, final.argmax(1)) or not final.isfinite().all():
        raise RuntimeError("reference predictions and captured final scores differ")
    if model.log6["v6"] != len(x) or any(
        log["err"] for log in (model.log6, model.cf.log, model.cp.log)
    ):
        raise RuntimeError("frozen reference failed a stage during the prefix")
    counters = torch.tensor([len(x), model.log6["rec"], model.ft.refits,
                             model.ft.steps, model.ft.relabels, model.cp.refits])
    torch.save(dict(native_format=torch.tensor(1), x=x, scores=final, counters=counters), output)


if __name__ == "__main__":
    main()
