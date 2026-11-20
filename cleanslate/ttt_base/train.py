"""ttt_base trainer: team C's recipe (ref/train.py: causal per-epoch alignment, session-homogeneous batches) for a wider
member zoo, optional pretrained trunks, writes a team C compatible package.

python cleanslate/ttt_base/train.py --people P01,... --out DIR --members shallow:0,fbc@hp7:1,eegnex:0 [--epochs 60]
Member = PRESET:SEED or PRESET@TRUNK:SEED. Presets below; TRUNK = runs/trunks/fbcnet_raw-{TRUNK}-s{SEED}.pt (Lane H / B6
pretraining of braindecode FBCNet; hp7 = 7 corpora, hbx = + B6's 81 people). The trunk must load (>= all but 4 keys) or fail.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ref"))
import train as T  # noqa: E402  (ref/train.py: load, Aligner)

ROOT = HERE.parents[1]
PRESETS = {  # arch, kw
    "shallow": ("shallow", None), "eegnet": ("eegnet", None), "fbnet": ("fbnet", None),
    "fbwide": ("fbnet", dict(m=8)),
    "fbc": ("bd", dict(name="FBCNet")), "fbc64": ("bd", dict(name="FBCNet", n_filters_spat=64)),
    "fbcst8": ("bd", dict(name="FBCNet", stride_factor=8)),
    "eegnex": ("bd", dict(name="EEGNeX")), "ifnet": ("bd", dict(name="IFNet")),
}


def generate_submission(out):
    """ref/submission.py + extras + BatchNorm generalisation; written to OUT/submission.py and imported as the module."""
    src = (HERE / "ref/submission.py").read_text()
    key = "ARCHS = dict(shallow=Shallow, eegnet=EEGNet, fbnet=FBNet)\n"
    assert src.count(key) == 1 and src.count("nn.BatchNorm2d") >= 3
    src = src.replace(key, key + (HERE / "archs.py").read_text())
    n = src.count("isinstance(m, nn.BatchNorm2d)")
    assert n == 3, n
    src = src.replace("isinstance(m, nn.BatchNorm2d)", "isinstance(m, nn.modules.batchnorm._BatchNorm)")
    k1, k2 = "rot=0.0, max_n=1024)", "            for net, opt, anc in zip(self.nets, self.opts, self.anchor):\n"
    assert src.count(k1) == 1 and src.count(k2) == 1
    src = src.replace(k1, "rot=0.0, max_n=1024, freeze_bd=0)")  # freeze_bd=1: braindecode members get BN refresh but no gradient steps
    src = src.replace(k2, k2 + '                if c["freeze_bd"] and isinstance(net, BD):\n                    continue\n')
    (out / "submission.py").write_text(src)
    spec = importlib.util.spec_from_file_location("ttt_base_submission", out / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_trunk(net, trunk, seed):
    f = ROOT / f"runs/trunks/fbcnet_raw-{trunk}-s{seed}.pt"
    sd = torch.load(f, map_location="cpu")["state"]
    own = net.net.state_dict()
    sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    assert len(sd) >= len(own) - 4, f"trunk {f.name}: {len(sd)} of {len(own)} keys match"
    net.net.load_state_dict(sd, strict=False)
    return len(sd), len(own)


def fit_member(S, preset, trunk, seed, al, y, args, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    arch, kw = PRESETS[preset]
    net = S.build(arch, kw).to(dev)
    moved = load_trunk(net, trunk, seed) if trunk else None
    assert (trunk is None) or arch == "bd", "trunks exist for braindecode FBCNet only"
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (len(y) // 48 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    yt = torch.tensor(y, device=dev)
    k = 0
    for ep in range(args.epochs):
        A, batches = al.epoch(rng)
        net.train()
        for b in batches:
            bt = torch.tensor(b, device=dev)
            xb = S.augment(A[bt], g, rot=args.rot) if args.aug else A[bt]
            loss = F.cross_entropy(net(xb), yt[bt], label_smoothing=args.ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if k < steps - 1:
                sched.step()
                k += 1
    net.eval()
    return net, arch, kw, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", default="shallow:0,eegnet:0,fbnet:0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--aug", type=int, default=1)
    ap.add_argument("--data", default=str(ROOT / "data/proteus_windows.npz"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S = generate_submission(out)
    T.S = S  # Aligner / loader use the package's own alignment code
    shrink = S.DEFAULT_TTT["shrink"]
    people = sorted(args.people.split(","))
    X, y, stream = T.load(people, args.data)
    al = T.Aligner(X, stream, dev, shrink)
    members, log = [], []
    for spec in args.members.split(","):
        name, seed = spec.rsplit(":", 1)
        preset, _, trunk = name.partition("@")
        t0 = time.time()
        net, arch, kw, moved = fit_member(S, preset, trunk or None, int(seed), al, y, args, dev)
        members.append(dict(arch=arch, kw=kw, seed=int(seed), name=name, state={k: v.cpu() for k, v in net.state_dict().items()}))
        log.append(dict(member=spec, sec=round(time.time() - t0, 1), params=sum(p.numel() for p in net.parameters()), trunk_keys=moved))
        print("member", log[-1], flush=True)
    torch.save(dict(members=members, ttt={**S.DEFAULT_TTT, "shrink": shrink}, people=people, args=vars(args)), out / "weights.pt")
    json.dump(dict(people=people, streams=sorted({s.split("|")[0] + "|" + s.split("|")[1] for s in stream}), args=vars(args),
                   members=log, rows=len(y)), open(out / "train.json", "w"), indent=1)


if __name__ == "__main__":
    main()
