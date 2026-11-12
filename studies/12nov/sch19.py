"""SCH19 (ledger s53): Scherer 2015 as extra training people on the 19 channels both caps really have.
Usage: python studies/12nov/sch19.py"""
import os, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
d = protocol.load(external=("scherer2015",)); interp = set(map(str, np.load(REPO + "/data/scherer2015/windows.npz", allow_pickle=True)["interpolated"]))
keep = np.array([i for i, c in enumerate(map(str, d["ch_names"])) if c not in interp]); print("channels kept", len(keep), [str(d["ch_names"][i]) for i in keep], flush=True)
d = dict(d, X=d["X"][:, keep]); T.BANDSETS["hp4"] = [b for b in T.BANDSETS["ov17"] if b[0] >= 4]
fe = T.Feats(d, T.parse("car_hp4_fs2_t0.1_le_k2")); F = fe.F[:, :fe.slices[-1].start]  # the slow block is dropped for everyone
y = d["y"]; ext = d["split"] == "external"; per = np.array([s.split("/")[-1] for s in d["subject"]]); st = d["stream"]; itf = d["interface"]
print("features", F.shape, "PROTEUS rows", int((~ext).sum()), "Scherer rows", int(ext.sum()), "Scherer people", sorted(set(per[ext])), flush=True)
NB = len(fe.C)


def causal(rows, idx):
    pri = [fe.prior(b, idx) for b in range(NB)]; d2p = [q[idx].mean() for q in fe.d2]; out = np.zeros((len(rows), F.shape[1]))
    for s in dict.fromkeys(st[rows]):
        m = st[rows] == s; out[m] = fe.causal(rows[m], pri, d2p, None)[:, :F.shape[1]]
    return out


def ba(r, pred):
    return 100 * np.mean([np.mean([np.mean(pred[(itf[r] == i) & (y[r] == c)] == c) for c in range(3)]) for i in np.unique(itf[r])])


def fit(idx, w=None):
    return make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)).fit(F[idx], y[idx], logisticregression__sample_weight=w)


P_people = sorted(set(per[~ext])); S_people = sorted(set(per[ext])); res = {k: [] for k in ("R19", "S19 w1.0", "S19 w0.3")}
for p in P_people:
    tr = np.flatnonzero(~ext & (per != p)); ex = np.flatnonzero(ext)
    sess = [np.flatnonzero(~ext & (st == s)) for s in dict.fromkeys(st[~ext & (per == p)])]
    Ft = [causal(r, tr) for r in sess]  # the causal prior comes from the PROTEUS training rows in every arm
    for name, idx, w in (("R19", tr, None), ("S19 w1.0", np.r_[tr, ex], None), ("S19 w0.3", np.r_[tr, ex], np.r_[np.ones(len(tr)), np.full(len(ex), 0.3)])):
        m = fit(idx, w); res[name].append(np.mean([ba(r, m.predict(f)) for r, f in zip(sess, Ft)]))
    print(f"  {p} " + " ".join(f"{k} {v[-1]:.1f}" for k, v in res.items()), flush=True)
for k, v in res.items():
    print(f"RESULT {k:9s} mean {np.mean(v):.2f} ({np.mean(v) - np.mean(res['R19']):+.2f} vs R19; {int((np.array(v) > np.array(res['R19']) + 1e-9).sum())} of 10 up) | " + " ".join(f"{p[1:]}:{x:.1f}" for p, x in zip(P_people, v)), flush=True)
allp = np.flatnonzero(~ext); m = fit(allp); a = []
for q in S_people:
    for s in dict.fromkeys(st[ext & (per == q)]):
        r = np.flatnonzero(st == s); a.append(ba(r, m.predict(causal(r, allp))))
print(f"RESULT PROTEUS -> Scherer: mean session BA {np.mean(a):.2f} (min {np.min(a):.1f}, max {np.max(a):.1f}, {len(a)} sessions)", flush=True)
a = []
for q in S_people:
    tr = np.flatnonzero(ext & (per != q)); m = fit(tr)
    for s in dict.fromkeys(st[ext & (per == q)]):
        r = np.flatnonzero(st == s); a.append(ba(r, m.predict(causal(r, tr))))
print(f"RESULT Scherer leave-one-person-out: mean session BA {np.mean(a):.2f} (min {np.min(a):.1f}, max {np.max(a):.1f})", flush=True)
