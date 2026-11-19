"""Clean reference train.py (harness contract). Switches via env CS_REF_EA / CS_REF_ADABN / CS_REF_PRIOR (default 1/1/0)."""
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data as D

HERE = Path(__file__).resolve().parent
EPOCHS, BATCH = 40, 64


def flag(name, default):
    v = os.environ.get(name, str(default))
    assert v in ("0", "1"), f"{name} must be 0 or 1, got {v}"
    return int(v)


def main():
    a = D.train_parser()
    cfg = dict(ea=flag("CS_REF_EA", 1), adabn=flag("CS_REF_ADABN", 1), prior=flag("CS_REF_PRIOR", 0), seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("sub", HERE / "submission.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    X, y, info = D.load_train(a.people, a.first_session_only, a.data)
    rng = np.random.default_rng(a.seed)
    if cfg["ea"]:  # causal EA over a random delivery order, as in full-session-shuffled streaming
        stream = np.char.add(info["person"], info["session"])
        for s in dict.fromkeys(stream):
            ea = sub.CausalEA()
            for r in rng.permutation(np.flatnonzero(stream == s)):
                X[r] = ea.apply(X[r])
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = sub.EEGNet().to(dev)
    Xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y).to(dev)
    w = torch.tensor(len(y) / (3 * np.bincount(y, minlength=3)), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * (len(y) // BATCH + 1))
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(y), device=dev)
        for i in range(0, len(y), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 4:
                continue
            loss = F.cross_entropy(net(Xt[b]), yt[b], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep % 10 == 9:
            print("epoch", ep + 1, "loss", float(loss), flush=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out / "weights.pt")
    json.dump(cfg, open(out / "config.json", "w"))
    shutil.copy(HERE / "submission.py", out / "submission.py")
    print("trained", cfg, len(y), "windows")


if __name__ == "__main__":
    main()
