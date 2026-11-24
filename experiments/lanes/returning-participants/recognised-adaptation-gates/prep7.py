"""ds007221g (Graz group, 14 people) per-person files for the B10-style replication: ext7/p{k}.npz with X (n,41,480), y (1 motor imagery = feet/left/right, 0 rest),
day (1 = ses-01, 2 = ses-02: two scenes / environments), run, onset, name, ch_names. Files may hold several people (windows_11-14). usage: prep7.py OUT_DIR WINDOWS_NPZ..."""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
k = 0
for f in sorted(sys.argv[2:]):
    with np.load(f, allow_pickle=False) as z:
        cls = [str(c) for c in z["classes"]]
        assert cls == ["feet", "left_hand", "rest", "right_hand"] and z["X"].shape[1:] == (41, 480), f
        for s in sorted(set(z["subject"].astype(str)), key=lambda v: int(v.split("/")[-1])):
            m = z["subject"].astype(str) == s
            ses = sorted(set(z["session"][m].astype(str)))
            assert ses == ["ses-01", "ses-02"], (s, ses)
            day = np.array([ses.index(v) + 1 for v in z["session"][m].astype(str)])
            y = (z["y"][m] != 2).astype(np.int8)
            run = np.array([int("".join(c for c in str(r) if c.isdigit())) for r in z["run"][m]], np.int64)
            np.savez(out / f"p{k}.npz", X=z["X"][m].astype(np.float32), y=y, day=day, run=run, onset=z["onset"][m].astype(np.float64),
                     name=np.array(s), ch_names=z["ch_names"])
            print(k, s, np.bincount(day)[1:].tolist(), np.bincount(y).tolist(), flush=True)
            k += 1
assert k == 14, k
