"""Why does CBraMod-C not learn? Frozen-feature spread and 150 fine-tune steps under fp32/bf16 and input scales."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]
import numpy as np, torch
import cbmod
from data import public
from model import masked_ce

root, ck = Path(sys.argv[1]), Path(sys.argv[2])
x, y, train, query, orders, names, prior = public(root, "P02", False)
X = torch.from_numpy(x[0]).cuda(); Y = torch.from_numpy(y).cuda()
print("input rms", float(X.pow(2).mean().sqrt()), "std per window", float(X.std((1, 2)).std()), flush=True)
pos = None
rng = np.random.default_rng(0)
for scale in (0.15,):
    cbmod.SCALE = scale
    m = cbmod.Member(ck).cuda().eval()
    pos = m.positions(names, "cuda")
    with torch.no_grad():
        ix = rng.choice(len(Y), 128, replace=False)
        f = torch.nn.functional.interpolate(X[ix], size=800, mode="linear")
        feats = m.body.net(f.clamp(-15, 15) * scale, return_features=True)["features"]
        pooled = feats.mean((1, 2))
        print(f"scale {scale}: feats absmean {float(feats.abs().mean()):.4f} pooled std-over-samples {float(pooled.std(0).mean()):.5f} "
              f"|mean| {float(pooled.mean(0).abs().mean()):.4f} nan {bool(torch.isnan(feats).any())}", flush=True)
    for amp in (True,):
        torch.manual_seed(0)
        m = cbmod.Member(ck).cuda().train(); opt = cbmod.split_optimizer(m); losses = []
        for s in range(400):
            ix = rng.choice(len(Y), 16, replace=False)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = masked_ce(m(X[ix], pos)[0], Y[ix], [0, 1, 2])
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step(); losses.append(float(loss))
        print(f"scale {scale} bf16={amp}: loss first20 {np.mean(losses[:20]):.3f} last50 {np.mean(losses[-50:]):.3f}", flush=True)
