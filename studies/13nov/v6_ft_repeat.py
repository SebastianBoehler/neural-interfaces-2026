"""Run-to-run spread of the packaged fine-tuning on one device (13 Nov, v6.1 GPU check at 1.02e-4 against 1e-4): the same
session streamed twice with both gates off through the same package; max |difference| of the fine-tuned posteriors and of
the outputs. On a GPU, atomic adds in convolution backward passes make training non-deterministic run to run.
Usage: python studies/13nov/v6_ft_repeat.py DEVICE PKG [N_SESSIONS]"""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import load_model  # noqa: E402


def run(m, X):
    c = copy.deepcopy(m); c.route.contrast.tau = c.route.identity.tau = np.inf; c.reset_state(); out, ft = [], []
    for x in X:
        out.append(c.step(np.nan_to_num(np.asarray(x, np.float32)))); f = c.trace["v6"].get("p_ft", {})
        ft.append(np.stack([f[k] for k in sorted(f)]) if f else np.full((4, 3), np.nan))
    return np.array(out), np.array(ft)


dev, pkg = sys.argv[1], Path(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
d = protocol.load(); m = load_model(pkg, None, "pkg", dev); torch.set_num_threads(1)
for s in list(dict.fromkeys(d["stream"]))[:n]:
    X = d["X"][d["stream"] == s]; o1, f1 = run(m, X); o2, f2 = run(m, X); k = ~np.isnan(f1[:, 0, 0])
    print(json.dumps(dict(session=s.split("/")[-1], ft_max_diff=float(np.abs(f1[k] - f2[k]).max()), out_max_diff=float(np.abs(o1 - o2).max()),
                          labels_differ=int((o1.argmax(1) != o2.argmax(1)).sum()))), flush=True)
