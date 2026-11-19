"""Time 20,083 unlabeled predictions using the exact package and worker loader."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from package import load_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as d:
        wave = torch.from_numpy(d["X"])
        names = d["ch_names"]
        stream = np.char.add(np.char.add(d["subject"], "/"), d["session"])
    rng = np.random.default_rng([71000, 0, 0])
    sessions = [rng.permutation(np.flatnonzero(stream == name)) for name in sorted(set(stream))]
    lengths = np.full(106, 20083 // 106)
    lengths[:20083 % 106] += 1
    sol = load_source(args.pkg / "submission.py").Solver()
    t0 = time.perf_counter()
    template = sol.load_model(dict(submission_dir=args.pkg, device="cuda", ch_names=names,
                                   n_chans=41, n_times=480, sfreq=120))
    torch.cuda.synchronize()
    load = time.perf_counter() - t0

    def deliver(current, rows, n):
        for i in np.resize(rows, n):
            label = current.predict(wave[i:i + 1].cuda())
            if label.shape != (1,) or label.dtype != torch.long or int(label) not in (0, 1, 2):
                raise RuntimeError("Invalid official prediction")

    start_warmup = time.perf_counter()
    warm = copy.deepcopy(template)
    deliver(warm, sessions[0], 48)
    del warm
    warmup = time.perf_counter() - start_warmup
    durations = []
    recognized_sessions = 0
    torch.cuda.reset_peak_memory_stats()
    for i, n in enumerate(lengths):
        start = time.perf_counter()
        current = copy.deepcopy(template)
        deliver(current, sessions[i % len(sessions)], n)
        durations.append(time.perf_counter() - start)
        recognized_sessions += current.person is not None
        del current
    torch.cuda.synchronize()
    result = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, sessions=len(lengths),
        windows=int(lengths.sum()), load_s=load, warmup_s=warmup, prediction_s=sum(durations),
        total_s=load + warmup + sum(durations), recognized_sessions=recognized_sessions,
        session_p95_s=float(np.percentile(durations, 95)), peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
