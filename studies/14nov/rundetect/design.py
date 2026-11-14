"""RUNDETECT design table (ledger s226, step 0; labels allowed, description only): per public session (all_loso; the
seen_session sessions are marked) the six runs in recording order with task, interface, length and class counts.
Writes runs/rundetect/run_design.txt. Usage: python studies/14nov/rundetect/design.py"""
import os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

with np.load(C.REPO + "/data/proteus_windows.npz") as w:
    task, run, itf = (w[k][C.IDX] for k in ("task", "run", "interface"))
seen = set(C.A.pl.B["idx|seen_session"].astype(int))
lines, kinds = [], Counter()
for s in C.SESSIONS:
    rows = np.flatnonzero(C.SES == s); tag = " seen_session" if seen & set(C.IDX[rows]) else ""
    lines.append(f"session {s} {C.PER[rows[0]]}{tag}: {len(rows)} windows")
    for u in dict.fromkeys(run[rows]):
        r = rows[run[rows] == u]; cc = np.bincount(C.Y[r], minlength=3); k = f"{task[r[0]]}"
        kinds[k, len(r), tuple(cc)] += 1
        lines.append(f"  run {u}: {task[r[0]]:16s} {itf[r[0]]:9s} length {len(r):3d} classes {cc[0]:2d}/{cc[1]:2d}/{cc[2]:2d}")
lines.append("pattern counts (task, length, class counts):")
lines += [f"  {k[0]:16s} {k[1]:3d} {'/'.join(map(str, k[2]))}: {n}" for k, n in sorted(kinds.items())]
open(C.REPO + "/runs/rundetect/run_design.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
