"""Fit the v6 eye net (studies/12nov/ocunet.py arm nomix, 60 epochs) on the permitted PROTEUS rows."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _aux_v6 import SMOKE, family_args, load_module, load_source, require_cuda, start, write_fit


def fit(args):
    require_cuda()
    ocu = load_module("ocunet", "studies/12nov/ocunet.py")
    d = load_source(args.people, args.first_session_only, args.data)
    traces = ocu.traces(d)  # running standardisation inside each source session
    n, seed, t0 = len(d["y"]), args.seed, start()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = ocu.Net().to(ocu.DEV)
    xt, yt = torch.from_numpy(traces), torch.from_numpy(d["y"])
    epochs = 1 if SMOKE else 60
    nb = int(np.ceil(n / 64))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * nb)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for b in range(nb):
            j = perm[b * 64:(b + 1) * 64]
            if len(j) < 8:
                sched.step()
                continue
            xb, yb = xt[j].to(ocu.DEV), torch.nn.functional.one_hot(yt[j], 3).float().to(ocu.DEV)
            xb = xb * (0.8 + 0.4 * torch.rand(len(j), 1, 1, generator=g).to(ocu.DEV)) + 0.05 * torch.randn(xb.shape, generator=g).to(ocu.DEV)
            opt.zero_grad()
            loss = -(yb * torch.log_softmax(net(xb), 1)).sum(1).mean()
            loss.backward()
            opt.step()
            sched.step()
    net = net.cpu().eval()
    with torch.no_grad():
        trace = torch.jit.trace(net, torch.from_numpy(traces[:4]), check_trace=False)
        x = torch.from_numpy(traces)
        pe, pt = torch.softmax(net(x).double(), 1), torch.softmax(trace(x).double(), 1)
    diff = float((pe - pt).abs().max())
    if diff >= 1e-5:
        raise ValueError(f"eye trace differs from eager by {diff}")
    if tuple(trace(x[:2]).shape) != (2, 3):
        raise ValueError("eye trace must emit 3 logits for (B,7,480)")
    trace.save(str(args.out / f"eye_s{seed}.pt"))
    write_fit(args.out, dict(seed=seed, epochs=epochs, rows=n, trace_vs_eager=diff, gpu=torch.cuda.get_device_name(),
                             train_acc=float((pe.argmax(1).numpy() == d["y"]).mean()), sec=round(time.time() - t0)))


if __name__ == "__main__":
    fit(family_args(__doc__, (0, 1, 2)))
