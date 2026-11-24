"""Lane E identity / function smoke: A.zip vs B.zip on N Scherer windows (unrecognised: refits from window 20).

python lane_e/smoke.py A.zip B.zip N DEVICE [--expect-identical]
Prints per arm the [v6] log fields (refits, ft nets, over) and the label / posterior differences. With
--expect-identical (default config vs cap250) any difference fails.
"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

a, b, n, dev = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
X, subject, session, ch = H.scherer(H.ROOT / "data/scherer2015/windows.npz")
rows = H.stream_rows(subject, session, n)
res = {}
for tag, zp in (("A", a), ("B", b)):
    with TemporaryDirectory() as tmp:
        model = H.load(zp, ch, dev, tmp)
        labels, outs, sec, m = H.run(model, X, rows, dev)
        res[tag] = (labels, outs)
        print(tag, Path(zp).name, f"{sec:.1f}s", "refits", m.ft.refits, "steps", m.ft.steps, "ft_nets", len(m.ft.member),
              "over", m.ft.over, "capped", m.ft.capped, "err", m.log6["err"], "labels", np.bincount(labels, minlength=3).tolist(), flush=True)
        del model, m
dl = int((res["A"][0] != res["B"][0]).sum())
dp = float(np.abs(res["A"][1] - res["B"][1]).max())
print(f"labels differ {dl}/{n}  max |out diff| {dp:.3g}")
if "--expect-identical" in sys.argv:
    assert dl == 0 and dp == 0.0, "default config is not bitwise cap250"
    print("IDENTICAL")
