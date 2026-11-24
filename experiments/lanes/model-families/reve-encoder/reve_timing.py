"""Warm per-window REVE-Base forward time (batch 1, 41 channels, 4 s at 120 Hz) for the packaging estimate."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model import Member


def main(checkpoint, out):
    device = torch.device("cuda")
    model = Member(Path(checkpoint)).to(device).eval()
    pos = torch.randn(1, 41, 3, device=device)  # timing only; positions do not change the cost
    x = torch.randn(1, 41, 480, device=device)
    res = dict(device=torch.cuda.get_device_name(0), params=sum(p.numel() for p in model.parameters()))
    for name, dtype in (("fp32", None), ("fp16", torch.float16)):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            for _ in range(30):
                model(x, pos)
            t = []
            for _ in range(300):
                torch.cuda.synchronize()
                s = time.perf_counter()
                model(x, pos)
                torch.cuda.synchronize()
                t.append(time.perf_counter() - s)
        res[name] = dict(median_ms=1000 * float(np.median(t)), p90_ms=1000 * float(np.quantile(t, 0.9)))
    res["fp32_state_bytes"] = 4 * res["params"]
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    print(res)


if __name__ == "__main__":
    main(*sys.argv[1:3])
