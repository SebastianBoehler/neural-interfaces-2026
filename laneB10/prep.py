"""NETBCI per-person external files for the B10 stress set: ext/p{k}.npz with X (n,41,480) f32, y (0 rest, 1 right_hand), day (1..4), run, onset.
usage: prep.py OUT_DIR WINDOWS_NPZ... ; people sorted by name over all inputs; day = rank of the session label within the person (asserted 4 days)."""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
parts = {}
for f in sys.argv[2:]:
    with np.load(f, allow_pickle=False) as z:
        assert z["X"].shape[1:] == (41, 480) and list(z["classes"]) == ["rest", "right_hand"], f
        for s in sorted(set(z["subject"])):
            m = z["subject"] == s
            parts[str(s)] = {k: z[k][m] for k in ("X", "y", "session", "run", "onset")}
            ch = z["ch_names"]
names = sorted(parts, key=lambda s: int(s.split("-")[-1]) if s.split("-")[-1].isdigit() else s)
assert len(names) == 19, names
for k, s in enumerate(names):
    p = parts[s]
    ses = sorted(set(p["session"].astype(str)))
    assert len(ses) == 4, (s, ses)
    day = np.array([ses.index(v) + 1 for v in p["session"].astype(str)])
    np.savez(out / f"p{k}.npz", X=p["X"].astype(np.float32), y=p["y"].astype(np.int8), day=day, run=np.array([int("".join(c for c in str(r) if c.isdigit())) for r in p["run"]], np.int64),
             onset=p["onset"].astype(np.float64), name=np.array(s), sessions=np.array(ses), ch_names=ch)
    print(k, s, ses, np.bincount(day)[1:].tolist(), np.bincount(p["y"]).tolist(), flush=True)
