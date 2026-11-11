"""How good can labels for PAST windows get without ground truth? (They are only used to fine-tune, so they may use
everything seen so far.) Compares, at the end of each session: the causal chain output, forward x backward smoothing
(the session stage run in both directions), and a balanced assignment inside each run (10 or 12 per class)."""
import importlib.util, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
REPO = __import__("os").path.abspath(__file__ + "/../../../.."); T = REPO + "/.weco/session-stage"
spec = importlib.util.spec_from_file_location("st", T + "/baseline.py"); st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
D = np.load(T + "/pack/data.npz", allow_pickle=True); W = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)
run_id = np.char.add(np.char.add(W["task"].astype(str), "|"), W["run"].astype(str)); p = "all_loso"; bn = "B2_pool"
ss, y, M, X, F = D[f"sess|{p}"], D[f"y|{p}"], D[f"mix|{bn}|{p}"], D[f"X|{p}"], D[f"F|{p}"]


def balance(P, runs):
    out = P.argmax(1).copy()
    for r in dict.fromkeys(runs):
        m = np.flatnonzero(runs == r); n = len(m); k = n // 3; slots = np.repeat(np.arange(3), k)
        if 3 * k != n:  # uneven run (BrainHero acquisition): leave the extra windows free
            slots = np.concatenate([slots, np.arange(3)[: n - 3 * k]])
        cost = -np.log(np.clip(P[m], 1e-9, 1))[:, slots]; ri, ci = linear_sum_assignment(cost); out[m[ri]] = slots[ci]
    return out


acc = {k: [] for k in ("causal", "smoothed", "causal + balance", "smoothed + balance")}; early = {k: [] for k in acc}; n_tot = 0
for s in np.unique(ss):
    r = np.flatnonzero(ss == s); fw = st.run_session(M[r], X[r], F[r]); bw = st.run_session(M[r][::-1], X[r][::-1], F[r][::-1])[::-1]
    mixr = M[r] / M[r].sum(1, keepdims=True); sm = fw * bw / np.clip(mixr, 1e-9, 1); sm /= sm.sum(1, keepdims=True)
    labs = {"causal": fw.argmax(1), "smoothed": sm.argmax(1), "causal + balance": balance(fw, run_id[r]), "smoothed + balance": balance(sm, run_id[r])}
    for k, l in labs.items():
        acc[k].append((l == y[r]).sum()); early[k].append((l[:30] == y[r][:30]).sum())
    n_tot += len(r)
ns = len(np.unique(ss))
for k in acc:
    print(f"{k:20s} label accuracy {sum(acc[k]) / n_tot:.4f} | first 30 windows {sum(early[k]) / (30 * ns):.4f}")
