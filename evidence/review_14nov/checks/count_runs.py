"""Read-only: run / interface / task structure per public session (data/proteus_windows.npz)."""
import sys, numpy as np
D = np.load(sys.argv[1], allow_pickle=True)
print("keys:", {k: (D[k].shape, str(D[k].dtype)) for k in D.files if k != "X"})
keys = D.files
sub = D["subject"] if "subject" in keys else D["person"]
ses = D["session"]; run = D["run"]; itf = D["interface"]; y = D["y"]
task = D["task"] if "task" in keys else None
onset = D["onset"] if "onset" in keys else None
for s in sorted(set(zip(sub.tolist(), ses.tolist()))):
    m = (sub == s[0]) & (ses == s[1])
    parts = []
    for r in dict.fromkeys(run[m].tolist()):
        mr = m & (run == r)
        parts.append(f"r{r}:{itf[mr][0]}{'/' + str(task[mr][0]) if task is not None else ''}:{int(mr.sum())}")
    print(s, int(m.sum()), " ".join(parts))
