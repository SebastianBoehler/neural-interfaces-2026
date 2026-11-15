"""s281 cost: per-window forward time of fb, fb_coral and fb_am under causal SessionBN (cost.py's harness imported, same loop). CORAL and amplitude mixing are
training-side only, so the three nets must have identical parameter counts and (within noise) identical time; the added net counts as one more fb forward.
Usage (Leonardo A100): python studies/14nov/newmember/cost_s.py cuda"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; spec = importlib.util.spec_from_file_location("cost", HERE / "cost.py"); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
CFG = {"fb": "fb_sbn_rot05", "fb_coral": "fb_sbn_rot05_coral", "fb_am": "fb_sbn_rot05_am"}
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(int(os.environ.get("NTHREADS", 1))); d = C.protocol.load(); res, par = {}, {}
print(json.dumps(dict(torch=torch.__version__, device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None)), flush=True)
for rep in range(3):  # three passes, the median over passes of the per-t median
    for t in (30, 110, 190):
        X = torch.randn(t, d["X"].shape[1], d["X"].shape[2], device=dev)
        for fam, cfg in CFG.items():
            n, nsb, npar = C.net(cfg, 2.0, d, dev); par[fam] = npar; res.setdefault((fam, t), []).append(1000 * C.bench(n, X, dev, 5 if dev == "cpu" else 20) / t)
ms = {f: float(np.mean([np.median(res[f, t]) for t in (30, 110, 190)])) for f in CFG}
assert len(set(par.values())) == 1, par
print(json.dumps(dict(ms_per_window=ms, params=par, extra_net_ms=ms["fb_coral"], extra_both_ms=ms["fb_coral"] + ms["fb_am"])), flush=True)
