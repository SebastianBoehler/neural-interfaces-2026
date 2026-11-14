"""Read-only count of people, sessions, cells, windows in the dev protocols (runs/replay_data/bases.npz)."""
import sys, numpy as np
B = np.load(sys.argv[1], allow_pickle=True)
print("keys:", [k for k in B.files if "|" in k and k.split("|")[0] in ("y", "cell", "person", "sess", "idx", "run", "interface")][:40])
for proto in ("all_loso", "seen_session"):
    y, cell, per, ss = B[f"y|{proto}"], B[f"cell|{proto}"], B[f"person|{proto}"], B[f"sess|{proto}"]
    print(f"\n{proto}: windows {len(y)}, people {len(np.unique(per))}, sessions {len(np.unique(ss))}, cells {len(np.unique(cell))}")
    for p in np.unique(per):
        m = per == p
        row = []
        for s in np.unique(ss[m]):
            ms = m & (ss == s)
            row.append(f"{s}: " + ", ".join(f"{c}={int((ms & (cell == c)).sum())}w/cls{np.bincount(y[ms & (cell == c)].astype(int), minlength=3).tolist()}" for c in np.unique(cell[ms])))
        print(f"  {p}: windows {int(m.sum())}, sessions {len(np.unique(ss[m]))}, cells {len(np.unique(cell[m]))} | " + " ; ".join(row))
