"""ocushuf.py standardises every session in the all_loso delivery order (line 10). Is that the order the seen_session
stages use for P04/P10 sessions 2-3? Compares runs/shuffle_perms.npz <scheme>|<seed>|all_loso with |seen_session per session."""
import os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); P = np.load(REPO + "/runs/shuffle_perms.npz")
w = np.load(REPO + "/data/proteus_windows.npz"); st = np.char.add(np.char.add(w["subject"].astype(str), "|"), w["session"].astype(str))
print("keys:", len(P.files), sorted({k.split("|")[0] + "|" + k.split("|")[2] for k in P.files}))
for sc, sd in (("within", 0), ("within", 1), ("full", 0), ("runs", 0)):
    a, s = P[f"{sc}|{sd}|all_loso"], P[f"{sc}|{sd}|seen_session"]
    for u in dict.fromkeys(st[s]):
        oa, os_ = a[st[a] == u], s[st[s] == u]; assert sorted(oa) == sorted(os_)
        ra = {g: k for k, g in enumerate(oa)}; pos = np.array([ra[g] for g in os_])  # position in the all_loso order of each window, in seen delivery order
        # share of windows that the eye net standardised with a prefix differing from the stage's delivered prefix
        same_prefix = np.mean([set(oa[:ra[g] + 1]) == set(os_[:k + 1]) for k, g in enumerate(os_)])
        print(f"{sc}{sd} {u.split('/')[-1]}: same order {bool((oa == os_).all())}; windows whose eye-net prefix equals the delivered prefix {same_prefix:.2f}; "
              f"rank corr of the two orders {np.corrcoef(pos, np.arange(len(pos)))[0, 1]:+.2f}")
