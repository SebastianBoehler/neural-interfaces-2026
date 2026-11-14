"""BAGCORE (ledger s223): core1 (car_ov17_fs2_t0.1_le_k2, scaler + LR C 0.3) trained on subsets of the training people.
Per fold of all_loso and seen_session: test features with the causal references of the FULL fold (priors from all
training rows; the references are test-time state, not a model choice), then
  full  one fit on all training rows (must reproduce the stored core1 member, correlation >= 0.99, checked in bagscore.py)
  l1_q  leave training person q out (one bag per training person)
  l2_j  NL2 random leave-two-people-out bags (seed 0)
Saves runs/bagging/bagcore.npz: <proto>|full -> (N, 3); <proto>|l1 -> (B1, N, 3); <proto>|l2 -> (B2, N, 3) (NaN off test),
plus per-fit seconds. Usage: [NL2=8] [NJ=4] [PROTOS=all_loso,seen_session] python studies/14nov/bagging/bagcore.py"""
import os, sys, time
import numpy as np
from joblib import Parallel, delayed
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

CFG, C = "car_ov17_fs2_t0.1_le_k2", 0.3
NL2, NJ = int(os.environ.get("NL2", 8)), int(os.environ.get("NJ", 4))
PROTOS = os.environ.get("PROTOS", "all_loso,seen_session").split(",")
T0 = time.time(); d = protocol.load(); fe = T.Feats(d, T.parse(CFG)); N = len(d["y"]); subj = d["subject"]
print(f"BAGCORE: core1 features {fe.F.shape} built ({time.time() - T0:.0f}s); NL2 {NL2} NJ {NJ}", flush=True)


def fit_predict(rows, Fte):
    t = time.time(); m = T.fit_head(f"lr{C}", fe.F[rows], fe.y[rows], fe.slices)
    return m.predict_proba(Fte), time.time() - t


OUT, SECS = {}, []
for proto in PROTOS:
    full = np.full((N, 3), np.nan); l1, l2 = None, np.full((NL2, N, 3), np.nan)
    for name, tr, te in protocol.folds(d, proto):
        tri = np.flatnonzero(tr); rows_te = np.flatnonzero(te)
        pri = [fe.prior(b, tri) for b in range(len(fe.C))]; d2p = [q[tri].mean() for q in fe.d2]; szp = fe.sz_prior(tri)
        Fte = np.zeros((len(rows_te), fe.F.shape[1]))
        for s in dict.fromkeys(d["stream"][rows_te]):
            k = d["stream"][rows_te] == s; Fte[k] = fe.causal(rows_te[k], pri, d2p, szp)
        people = sorted(set(subj[tri])); rng = np.random.default_rng(0)
        pairs = [rng.choice(len(people), 2, replace=False) for _ in range(NL2)]
        jobs = [tri] + [tri[subj[tri] != q] for q in people] + [tri[~np.isin(subj[tri], [people[i] for i in p])] for p in pairs]
        res = Parallel(n_jobs=NJ, prefer="threads")(delayed(fit_predict)(r, Fte) for r in jobs)
        if l1 is None:
            l1 = np.full((len(people), N, 3), np.nan)
        full[rows_te] = res[0][0]
        for i in range(len(people)):
            l1[i, rows_te] = res[1 + i][0]
        for j in range(NL2):
            l2[j, rows_te] = res[1 + len(people) + j][0]
        SECS += [r[1] for r in res]
        print(f"  {proto} {name}: {len(jobs)} fits, mean {np.mean([r[1] for r in res]):.0f}s per fit ({time.time() - T0:.0f}s)", flush=True)
    OUT.update({f"{proto}|full": full, f"{proto}|l1": l1.astype(np.float32), f"{proto}|l2": l2.astype(np.float32)})
os.makedirs(REPO + "/runs/bagging", exist_ok=True)
np.savez_compressed(REPO + "/runs/bagging/bagcore.npz", secs=np.array(SECS), **OUT)
print("saved runs/bagging/bagcore.npz", f"{time.time() - T0:.0f}s")
