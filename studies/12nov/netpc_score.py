"""Score the net people curve (ledger s56) and the eye-movement branch alone (s58 step 3). Usage: python studies/12nov/netpc_score.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
SC = env.A.SC; D = REPO + "/runs/netaug/probs"; pe = lambda P: SC.person(P, "all_loso")  # noqa: E731
for arch in ("fb", "fbl"):
    m = {k: np.mean([pe(np.load(f)["proba"]).mean() for f in glob.glob(f"{D}/pc_{arch}_k{k}_d*-all_loso-s0.npz")]) for k in (3, 5, 7)}
    m[9] = pe(np.load(f"{D}/na_{arch}_twin-all_loso-s0.npz")["proba"]).mean(); ks = np.array(sorted(m), float); v = np.array([m[k] for k in sorted(m)])
    print(f"NETPC {arch}: " + " ".join(f"k={int(k)}: {x:.2f}" for k, x in zip(ks, v)) + f" | per doubling {np.polyfit(np.log2(ks), v, 1)[0]:+.2f}; 7 -> 9 {v[-1] - v[-2]:+.2f}")
D2 = REPO + "/runs/ocunet/probs"
for arm in ("mix", "nomix"):
    zs = [np.load(f) for f in sorted(glob.glob(f"{D2}/oc_{arm}-all_loso-s*.npz"))]; P = np.mean([z["proba"] for z in zs], 0)
    print(f"OCUNET {arm}: single seeds " + " ".join(f"{pe(z['proba']).mean():.2f}" for z in zs) + f" | mean of seeds {pe(P).mean():.2f} | per person " + " ".join(f"{q[1:]}:{x:.0f}" for q, x in zip(SC.PEOPLE, pe(P))))
