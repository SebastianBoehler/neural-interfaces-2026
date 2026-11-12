"""PROFILE (12 Nov research window): where the v5.1 replay chain loses, by person, interface, run type and session
phase (new people LOSO, returning people seen protocol), B2_pool. Usage: python studies/12nov/profile.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102
D = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); IT, TK = D["interface"].astype(str), D["task"].astype(str)
REC.clear(); out0, _ = ns["system4"]("B2_pool", "max", 10); out, _ = staged(out0)
for proto, tag in (("all_loso", "new people (LOSO)"), ("seen_session", "returning (seen)")):
    idx = A.pl.B[f"idx|{proto}"].astype(int); P = out[proto]; y = O.y[idx]; ok = P.argmax(1) == y; per, ss = O.PER[idx], A.pl.B[f"sess|{proto}"]
    pos = np.zeros(len(idx))
    for s in np.unique(ss):
        r = np.flatnonzero(ss == s); pos[r] = np.arange(len(r)) / len(r)
    print(f"== {tag}: window accuracy {100 * ok.mean():.1f}%")
    print("   by interface: " + " | ".join(f"{i} {100 * ok[IT[idx] == i].mean():.1f}" for i in np.unique(IT[idx])))
    print("   by run type:  " + " | ".join(f"{t} {100 * ok[TK[idx] == t].mean():.1f}" for t in np.unique(TK[idx])))
    print("   by fifth:     " + " | ".join(f"{100 * ok[(pos >= q / 5) & (pos < (q + 1) / 5)].mean():.1f}" for q in range(5)))
    print("   by class:     " + " | ".join(f"{c} {100 * ok[y == k].mean():.1f}" for k, c in enumerate(("MI", "SUB", "WORD"))))
    print("   by person:    " + " ".join(f"{q[1:]}:{100 * ok[per == q].mean():.0f}" for q in np.unique(per)))
    cm = np.zeros((3, 3))
    for a, b in zip(y, P.argmax(1)):
        cm[a, b] += 1
    print("   confusion (rows true MI/SUB/WORD, %): " + str((100 * cm / cm.sum(1, keepdims=True)).round().astype(int).tolist()))
