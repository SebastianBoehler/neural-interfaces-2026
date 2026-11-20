"""Stream a team A MANAS fold package under ttt_base's orders and keep the softmax per window (late-fusion member).

python cleanslate/ttt_base/manas_stream.py --pkg DIR --held P01 --orders 0,1 --out FILE.npz
The package's own submission.py is used unchanged except one added line that stores the logits (`self.last`) before the argmax.
Same rows/order/session semantics as ref/stream_eval.py (deepcopy + reset_state per session, one window per call).
"""
import argparse
import copy
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
OLD = "return logits.argmax(-1)"


def load(pkg, ch_names, dev, tmp):
    src = (Path(pkg) / "submission.py").read_text()
    assert src.count(OLD) == 1, "package predict changed; cannot tap the logits"
    Path(tmp).mkdir(parents=True, exist_ok=True)
    patched = Path(tmp) / "submission.py"
    patched.write_text(src.replace(OLD, "self.last = logits.float().softmax(-1)[0]\n        " + OLD))
    spec = importlib.util.spec_from_file_location("manas_pkg", patched)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = dict(submission_dir=Path(pkg), device=dev, ch_names=list(ch_names), n_chans=len(ch_names), n_times=480,
                n_classes=3, sfreq=120.0, chs_info=[{"ch_name": c} for c in ch_names])
    return mod.Solver().load_model(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--orders", default="0,1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/proteus_windows.npz"))
    ap.add_argument("--perms", default=str(ROOT / "runs/shuffle_perms.npz"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data)
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    stream = np.char.add(np.char.add(person, "|"), d["session"])
    query = np.flatnonzero(person == a.held)
    model = load(a.pkg, d["ch_names"], dev, Path(a.out).with_suffix(".tmp"))
    out = {}
    for k in map(int, a.orders.split(",")):
        with np.load(a.perms) as p:
            rows = p[f"full|{k}|all_loso"]
        rows = rows[np.isin(rows, query)]
        assert np.array_equal(np.sort(rows), query)
        lab, post, sec = [], [], []
        for st in dict.fromkeys(stream[rows]):
            m = copy.deepcopy(model)
            m.reset_state()
            for r in rows[stream[rows] == st]:
                t0 = time.perf_counter()
                y = m.predict(torch.as_tensor(d["X"][r:r + 1]))
                if dev == "cuda":
                    torch.cuda.synchronize()
                sec.append(time.perf_counter() - t0)
                lab.append(int(y[0]))
                post.append(m.last.detach().cpu().numpy())
        out[f"rows|{k}"], out[f"M|{k}|label"], out[f"M|{k}|post"], out[f"M|{k}|sec"] = rows, np.array(lab), np.array(post), np.array(sec)
        print("PASS", a.held, k, round(float((np.array(lab) == d["y"][rows]).mean()), 4), flush=True)
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
