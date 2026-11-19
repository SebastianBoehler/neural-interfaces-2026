"""Lane B4: per-window forward time of one StreamBN-traced SessionBN net per family (batch 1, one window per call, as the worker streams).
python laneB4/timing.py OUT.json   (run as a Slurm job on an RTX-class GPU; untrained weights: architecture and state buffers decide the time)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
import session_bn  # noqa: E402

sys.path.insert(0, str(ROOT / "laneB4"))
from lopo_b4 import FAM  # noqa: E402

d = protocol.load(())
x = torch.from_numpy(np.asarray(d["X"][:1], dtype=np.float32))
res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, input=list(x.shape))
for fam, (cfg, k, _) in FAM.items():
    run_cv.SEED[0] = 0
    net = session_bn.make(cfg, d, 0)
    sn = session_bn.to_stream(net.net, k)
    dev = torch.device("cuda")
    ts = torch.jit.trace(sn, x, check_trace=False).to(dev).eval()
    xd = x.to(dev)
    with torch.no_grad():
        for _ in range(100):
            ts(xd)
        torch.cuda.synchronize()
        t = []
        for _ in range(1000):
            t0 = time.perf_counter(); ts(xd); torch.cuda.synchronize(); t.append(time.perf_counter() - t0)
    res[fam] = dict(ms_median=1e3 * float(np.median(t)), ms_mean=1e3 * float(np.mean(t)), ms_p95=1e3 * float(np.quantile(t, .95)),
                    params=sum(p.numel() for p in sn.parameters()))
    print(fam, res[fam], flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=2) + "\n")
