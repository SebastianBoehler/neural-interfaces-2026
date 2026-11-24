"""Revisit (Opus, 20-21 Nov): stream one held person's Team C package under a named shuffle_perms key and keep posteriors.

python revisit/stream_tc.py --root TC_RELEASE --pkg DIR --held P01 --key within|0|all_loso --variants A1,A2 --out FILE.npz
Same protocol as cleanslate/teamC/stream_eval.py (deepcopy + reset_state per session, one window per predict); only the
order key is free, so the chain's within0 order (== shuffle_perms 'within|0|all_loso') can be streamed. Output keys:
rows (delivery order of this person), V|post (n,3) and V|label in that order, key, held.
Zero-change smoke: key full|0|all_loso must reproduce cleanslate/teamC C1 eval_<held>.npz 'V|0|post'."""
import argparse
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    for k in ("root", "pkg", "held", "key", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--variants", default="A1,A2")
    a = ap.parse_args()
    root = Path(a.root)
    sys.path.insert(0, str(root / "cleanslate/teamC"))
    import stream_eval as SE  # noqa: E402
    perms = root / "runs/shuffle_perms.npz"
    assert hashlib.sha256(perms.read_bytes()).hexdigest() == SE.ORDER_SHA, "order archive changed"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(root / "data/proteus_windows.npz")
    X = d["X"]  # decompress once (indexing the NpzFile re-reads the whole array per window)
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    stream = np.char.add(np.char.add(person, "|"), d["session"])
    query = np.flatnonzero(person == a.held)
    with np.load(perms) as p:
        rows = p[a.key]
    rows = rows[np.isin(rows, query)]
    assert np.array_equal(np.sort(rows), query)
    out = dict(rows=rows, key=a.key, held=a.held)
    for v in a.variants.split(","):
        model = SE.load_model(a.pkg, dev, SE.VARIANTS[v])
        lab, post = [], []
        for st in dict.fromkeys(stream[rows]):
            m = copy.deepcopy(model)
            m.reset_state()
            for r in rows[stream[rows] == st]:
                y = m.predict(torch.as_tensor(X[r:r + 1]))
                assert y.shape == (1,)
                lab.append(int(y[0]))
                post.append(m.last.detach().cpu().numpy())
        out[f"{v}|post"], out[f"{v}|label"] = np.array(post), np.array(lab)
        print("PASS", a.held, a.key, v, round(float((np.array(lab) == d["y"][rows]).mean()), 4), flush=True)
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
