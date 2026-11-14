"""shuffle.py defines a run as task|run (line 27) and builds 'runs'/'within' blocks from it. What is `task`: the interface or the
class? Are a session's blocks contiguous in recording order, and does a delivery order carry label information (label runs)?"""
import os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); w = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)
print("fields:", [k for k in w.files if k != "X"])
task, run, y = w["task"].astype(str), w["run"].astype(str), w["y"].astype(int); itf = w["interface"].astype(str) if "interface" in w.files else None
st = np.char.add(np.char.add(w["subject"].astype(str), "|"), w["session"].astype(str))
print("task values:", dict(zip(*np.unique(task, return_counts=True))), "| interface values:", None if itf is None else dict(zip(*np.unique(itf, return_counts=True))))
print("task x y crosstab:", {t: np.bincount(y[task == t], minlength=3).tolist() for t in np.unique(task)})
if itf is not None:
    print("task == interface on every window:", bool((task == itf).all()))
RUN = np.char.add(task, np.char.add("|", run)); nb, contig, sizes = [], [], []
for s in dict.fromkeys(st):
    r = np.flatnonzero(st == s); u = list(dict.fromkeys(RUN[r])); nb.append(len(u))
    for b in u:
        rows = r[RUN[r] == b]; contig.append(bool((np.diff(rows) == 1).all())); sizes.append(len(rows))
print(f"sessions {len(nb)}; blocks per session {min(nb)}-{max(nb)}; blocks contiguous in recording order {np.mean(contig):.3f}; block size {min(sizes)}-{max(sizes)} (median {int(np.median(sizes))})")
P = np.load(REPO + "/runs/shuffle_perms.npz")
def same_next(order):  # share of consecutive delivered windows (same session) with the same label; chance is about 1/3
    a = order[:-1]; b = order[1:]; m = st[a] == st[b]; return float(np.mean(y[a][m] == y[b][m]))
print("P(next delivered window has the same label): recording", round(same_next(np.arange(len(y))), 3), "|", {f"{sc}{sd}": round(same_next(P[f"{sc}|{sd}|all_loso"]), 3) for sc in ("within", "full", "runs") for sd in (0, 1)})
