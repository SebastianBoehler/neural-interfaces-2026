"""Run-to-run spread of a fine-tuning package's scores on one GPU (coordinator, 13 Nov): every public session streamed R
times with both gates forced off (the unrecognised path, where fine-tuning runs) through the same package; per session
the balanced accuracy of each repeat, the spread (max - min) and the windows whose label differs between repeats; the
dev score (mean of session balanced accuracies) per repeat and its spread. In-sample members (they saw these sessions).
Usage: python studies/13nov/v6_repeat_ba.py DEVICE PKG [R]"""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import load_model  # noqa: E402


def run(m, X):
    c = copy.deepcopy(m); c.route.contrast.tau = c.route.identity.tau = np.inf; c.reset_state()
    out = np.array([c.step(np.nan_to_num(np.asarray(x, np.float32))) for x in X]); assert c.log6["err"] == 0, c.log6
    return out


def bacc(y, p):
    return float(np.mean([np.mean(p[y == k] == k) for k in np.unique(y)]))


dev, pkg = sys.argv[1], Path(sys.argv[2]); R = int(sys.argv[3]) if len(sys.argv) > 3 else 5
d = protocol.load(); m = load_model(pkg, None, "pkg", dev); torch.set_num_threads(1); B = []
for s in dict.fromkeys(d["stream"]):
    r = np.flatnonzero(d["stream"] == s); y = d["y"][r]; O = [run(m, d["X"][r]) for _ in range(R)]
    L = np.stack([o.argmax(1) for o in O]); ba = [bacc(y, l) for l in L]; B.append(ba)
    print(json.dumps(dict(session=s.split("/")[-1], bacc=[round(b, 4) for b in ba], spread=round(max(ba) - min(ba), 4),
                          windows_label_varies=int((L != L[0]).any(0).sum()), max_out_diff=float(max(np.abs(o - O[0]).max() for o in O)))), flush=True)
B = np.array(B); score = B.mean(0)
print("SUMMARY", json.dumps(dict(package=str(pkg), repeats=R, dev_score_per_repeat=[round(x, 5) for x in score], dev_score_spread=round(float(score.max() - score.min()), 5),
                                 sessions_with_varying_ba=int((B.max(1) - B.min(1) > 0).sum()), max_session_spread=round(float((B.max(1) - B.min(1)).max()), 4))), flush=True)
