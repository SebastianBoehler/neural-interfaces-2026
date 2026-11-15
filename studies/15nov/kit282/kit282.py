"""§282 step 1: start-kit REVE probe (solvers/reve_probe.py REVEEncoder + LinearProbe(RidgeClassifier), imported
unchanged from the submodule, nothing written there) on our dev windows, LOSO over the 10 PROTEUS people, person-equal BA.
Also: kit MeanLogReg, and our §71 embeddings (runs/fmemb/reve.npz) through the kit's ridge head and the kit's features
through our §71 head (z-score within session + LogReg C=0.1). Usage: scripts/run_capped.sh LOG python studies/15nov/kit282/kit282.py"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ["HF_HUB_OFFLINE"] = "1"  # never download
sys.path[:0] = [str(ROOT / "2026-competition/tracks/bci_decoding"), str(Path(__file__).parent),
                str(ROOT / "experiments")]
import numpy as np, torch
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import protocol
from kit_reve import REVEEncoder  # the shipped encoder, unmodified

d = protocol.load(); N = len(d["y"]); assert N == 2655 and (d["split"] != "external").all()
per, y = d["subject"], d["y"]
meta = dict(n_times=480, sfreq=120.0, n_chans=41, ch_names=[str(c) for c in d["ch_names"]], device="cpu")  # 480 -> 800 @200 Hz
OUT = ROOT / "runs/kit282"; OUT.mkdir(parents=True, exist_ok=True)
f = OUT / "kit_feats.npy"
if f.exists():
    F = np.load(f)
else:
    enc, t0, F = REVEEncoder(meta), time.time(), []
    for i in range(0, N, 32):
        F.append(enc.encode(torch.as_tensor(d["X"][i:i + 32])).mean(1).float().numpy())  # LinearProbe epoched pooling
        if i % 640 == 0: print(i, round(time.time() - t0), "s", flush=True)
    F = np.concatenate(F); np.save(f, F)
print("kit feats", F.shape, flush=True)

def sess_z(E):
    Z = np.empty(E.shape)
    for s in dict.fromkeys(d["stream"]):
        r = d["stream"] == s; Z[r] = (E[r] - E[r].mean(0)) / (E[r].std(0) + 1e-9)
    return Z

def loso(E, mk, z=False):
    E = sess_z(E) if z else E
    a = [bas(y[per == p], mk().fit(E[per != p], y[per != p]).predict(E[per == p])) for p in sorted(set(per))]
    return round(100 * float(np.mean(a)), 2), [round(100 * float(x), 1) for x in a]

ridge = lambda: make_pipeline(StandardScaler(), RidgeClassifier())
logr = lambda: LogisticRegression(C=0.1, max_iter=3000)
z = np.load(ROOT / "runs/fmemb/reve.npz", allow_pickle=True)
res = {
 "kit_feats + kit ridge head (AS SHIPPED)": loso(F, ridge),
 "kit_feats + our §71 head (sess-z, LogReg C=.1)": loso(F, logr, True),
 "our emb_raw (last, per-window z) + kit ridge head": loso(z["emb_raw"], ridge),
 "our emb_raw (last) + our head": loso(z["emb_raw"], logr, True),
 "our emb (last, cea) + our head": loso(z["emb"], logr, True),
 "our emb_mid (§71 headline layer, cea) + our head": loso(z["emb_mid"], logr, True),
 "our emb_rand + our head": loso(z["emb_rand"], logr, True),
 "kit MeanLogReg (mean over time)": loso(d["X"].mean(-1), lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
}
print("corr kit_feats vs our emb_raw (mean per-dim abs corr):",
      float(np.nanmean([abs(np.corrcoef(F[:, j], z["emb_raw"][:, j])[0, 1]) for j in range(F.shape[1])])))
for k, (m, a) in res.items(): print(f"{m:6.2f}  {k}  {a}")
json.dump(res, open(OUT / "result.json", "w"), indent=1)
