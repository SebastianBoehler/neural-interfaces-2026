"""CBraMod on an easy external task (Shin2017b rest vs subtraction): which setting learns? 300 steps each."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "laneC")]
import numpy as np, torch
import cbmod
from data import external_source
from model import augment, masked_ce

root, ck = Path(sys.argv[1]), Path(sys.argv[2])
x, y, motor, names, people, pins = external_source(root, "shin2017b", False)
print("rows", len(y), "classes", np.unique(y), "rms", float(np.sqrt((x ** 2).mean())), "chans", len(names), flush=True)
X = torch.from_numpy(x).cuda(); Y = torch.from_numpy(y).cuda()
valid = np.unique(y).tolist()
rng = np.random.default_rng(0)
cases = {"base lr1e-4/1e-3 scale.15": (1e-4, 1e-3, 0.15, True, False),
         "scale 1.0": (1e-4, 1e-3, 1.0, True, False),
         "frozen body (head only)": (0.0, 1e-3, 0.15, True, False),
         "body 1e-5": (1e-5, 1e-3, 0.15, True, False),
         "fp32": (1e-4, 1e-3, 0.15, False, False),
         "no dropout": (1e-4, 1e-3, 0.15, True, True)}
for name, (lrb, lrh, scale, amp, nodrop) in cases.items():
    cbmod.SCALE = scale
    torch.manual_seed(0); m = cbmod.Member(ck).cuda().train()
    if nodrop:
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout): mod.p = 0.0
    pos = m.positions(names, "cuda")
    opt = cbmod.split_optimizer(m); opt.param_groups[0]["lr"] = lrb; opt.param_groups[1]["lr"] = lrh
    if lrb == 0.0:
        for p in m.body.net.parameters(): p.requires_grad_(False)
    losses, accs = [], []
    for s in range(300):
        ix = rng.choice(len(Y), 16, replace=False)
        opt.zero_grad()
        xb, pb = augment(X[ix], pos, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lg = m(xb, pb)[0]; loss = masked_ce(lg, Y[ix], valid)
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 5); opt.step()
        losses.append(float(loss)); accs.append(float((lg[:, valid].argmax(1).cpu().numpy() == np.searchsorted(valid, Y[ix].cpu().numpy())).mean()))
    print(f"{name}: loss {np.mean(losses[:30]):.3f} -> {np.mean(losses[-60:]):.3f}  acc {np.mean(accs[-60:]):.3f}", flush=True)
