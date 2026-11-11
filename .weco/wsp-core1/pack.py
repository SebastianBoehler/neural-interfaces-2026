"""Build the compact evaluation pack for the Weco search over core1 head edits (from runs/wsp/cache)."""
import os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.argv[0], REPO + "/runs/wsp/cache"]
sys.path.insert(0, REPO + "/experiments")
import wsp_core as W  # noqa: E402
OUT = os.path.dirname(os.path.abspath(__file__)) + "/pack"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
ch_i = np.concatenate([np.tile(W.IU[0], W.NB), np.full(W.D - W.NB * 861, -1)]); ch_j = np.concatenate([np.tile(W.IU[1], W.NB), np.full(W.D - W.NB * 861, -1)])
np.savez(OUT + "/meta.npz", block=W.BLOCK, band=np.where(W.SLOW, -1, W.BAND), win=np.where(W.SLOW, -1, W.WIN), diag=W.DIAG, slow=W.SLOW,
         ch_i=ch_i, ch_j=ch_j, bands=W.BANDS, ch_names=np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)["ch_names"])
for nm in W.FOLDS:
    f = W.load(nm); o = f["out"]; ho, ins, co = W.inner_stats(f); people = list(f["inner"]); pick = rng.choice(len(people), 3, replace=False)
    Zin, qid = [], []
    for k in pick:
        m = f["inner"][people[k]]; te, _ = W.inner_rows(f, people[k]); te = rng.choice(te, min(100, len(te)), replace=False)
        Zin.append(((W.F[te] - m["mean"]) / m["scale"]).astype(np.float32)); qid += [k] * len(te)
    np.savez(f"{OUT}/{nm}.npz", rows=f["rows"], Zc=((f["Fc"] - o["mean"]) / o["scale"]).astype(np.float32), W=(o["coef"] - o["coef"].mean(0)).astype(np.float32),
             b=o["b"], inner=co.astype(np.float32), ho=ho.astype(np.float32), ins=ins.astype(np.float32), Zin=np.concatenate(Zin), qid=np.array(qid))
    print("packed", nm, flush=True)
