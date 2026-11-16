"""Compare unchanged control and CPU-resource pilot; compute no metric."""
import os
import sys
from pathlib import Path
import numpy as np

root = Path(os.environ["NI26_WS"]).resolve(strict=True)
sys.path.insert(0, str(root / "studies/14nov/rundetect"))
import package as K

bases = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")
for order in ("within0", "full0"):
    with np.load(root / f"runs/chainout/s281_v66p_control_{order}.npz") as a, np.load(root / f"runs/chainout/s281_v66p_geo_zero_{order}.npz") as b:
        if set(a.files) != set(b.files):
            raise ValueError(f"{order}: chain keys differ")
        for key in a.files:
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"{order}: chain {key} differs")
        lat = K.latch(order)
        cache = K.J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
        for base in bases:
            raw0, raw1 = a[f"{base}|V6Fr"].astype(np.float64), b[f"{base}|V6Fr"].astype(np.float64)
            if not np.array_equal(raw0, raw1):
                raise ValueError(f"{order}/{base}: raw differs")
            filt0 = K.gate(raw0, K.J.filt_joint(raw0, cache, "urn"), order, lat)
            filt1 = K.gate(raw1, K.J.filt_joint(raw1, cache, "urn"), order, lat)
            if not np.array_equal(filt0, filt1):
                raise ValueError(f"{order}/{base}: filtered differs")
    print(f"{order}: all raw and independently recomputed filtered arrays exact", flush=True)
