"""Lane T: team C's test-time-training recipe as a drop-in replacement of cap250's in-session refit (the r15c250 stream).

`make_run_stream(arm)` returns a function with the signature of friday/causal_replay/within0_warm.run_stream
(stream, model, x, target) -> (dict(base, r15, r15c250), steps). It replays ftrel.stream's r15 schedule (first refit at
window 20, every 10, refits stop at the first refit age >= 250, the last net keeps predicting) and replaces only the
refit objective:
  light : same optimiser (AdamW 1e-3, 5 steps), same stock strong view and KL(clean || strong) lam 1; hard weighted CE
          replaced by soft CE to the Sinkhorn-balanced chain posterior + information maximisation (0.3).
  heavy : 20 steps, per-window strong views (shift, scale, channel drop, relative noise), loss on <=64 random prefix
          windows per step (the forward runs over the whole prefix, SessionBN is causal in window order), L2 anchor to the
          source weights (1e-2), same targets.
Targets are the chain's own per-refit relabelled posteriors (target[:, 3:3+3K]); nothing else is read, no labels."""
import math

import numpy as np
import torch

ARMS = {
    "light": dict(steps=5, lam_im=0.3, batch=0, anchor=0.0, views="stock", balance=True),
    "heavy": dict(steps=20, lam_im=0.3, batch=64, anchor=1e-2, views="pw", balance=True),
    # fitting forms registered Fri 20 Nov ~09:40 (evidence/wave_20nov/laneT_ttt_cap250.md): heavy objective at about stock step cost
    "lite5": dict(steps=5, lam_im=0.3, batch=64, anchor=1e-2, views="pw", balance=True),
    "sparse20": dict(steps=20, lam_im=0.3, batch=64, anchor=1e-2, views="pw", balance=True, every=40),
    "light_nobal": dict(steps=5, lam_im=0.3, batch=0, anchor=0.0, views="stock", balance=False),
}
EVERY, T0, CAP, LR, LAM_KL = 10, 20, 250, 1e-3, 1.0


def sinkhorn(P, iters=10):
    Q = P.clamp_min(1e-8)
    for _ in range(iters):
        Q = Q / Q.sum(0, keepdim=True)
        Q = Q / Q.sum(1, keepdim=True)
    return Q


def aug_pw(X, g, shift=15, scale=0.15, drop=0.1, noise=0.2):
    """independent strong view per window: circular shift, amplitude scale, channel drop, noise at a share of the
    channel sd (generator g on X's device)"""
    B, C, T = X.shape
    s = torch.randint(-shift, shift + 1, (B, 1, 1), generator=g, device=X.device)
    idx = (torch.arange(T, device=X.device)[None, None] + s) % T
    Xs = torch.gather(X, 2, idx.expand(B, C, T))
    Xs = Xs * torch.exp(scale * torch.randn(B, 1, 1, generator=g, device=X.device))
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g, device=X.device)
    keep = (torch.rand(B, C, 1, generator=g, device=X.device) > drop).float()
    return Xs * keep


def make_run_stream(arm):
    cfg = ARMS[arm]

    def run_stream(stream, model, x, target):
        G = stream.__globals__
        T7, F, copy = G["T7"], G["F"], G["copy"]
        net, n, dev = model.net.eval(), len(x), x.device
        K = (target.shape[1] - 4) // 3
        R = target[:, 3:3 + 3 * K].reshape(n, K, 3)
        assert (target[:, -1] == np.arange(n)).all()
        with torch.no_grad():
            base = torch.softmax(net(x)[:, :3].double(), 1).cpu().numpy()
        out = base.copy()
        torch.manual_seed(0)
        gen = torch.Generator(device=dev).manual_seed(0)
        net2, steps, theta0 = None, 0, None
        every = cfg.get("every", EVERY)  # the relabelled-target grid stays 10 wide: index (t - T0) // EVERY is exact for 40 too
        assert every % EVERY == 0
        for t in range(T0, n, every):
            if t >= CAP:  # first refit at or after the cap: keep the last net from here on (ftrel's r15c250)
                with torch.no_grad():
                    for tt in range(t, n, every):
                        hh = min(n, tt + every)
                        out[tt:hh] = torch.softmax(net2(x[:hh])[:, :3].double(), 1).cpu().numpy()[tt:hh]
                break
            P = R[:t, (t - T0) // EVERY]
            assert np.isfinite(P).all(), t
            P = torch.from_numpy(P).to(dev).float()
            Q = sinkhorn(P) if cfg["balance"] else P
            if net2 is None:
                net2 = copy.deepcopy(net).eval()
                theta0 = [p.detach().clone() for p in T7.params_of(net2, "all")]
            ps = T7.params_of(net2, "all")
            opt = torch.optim.AdamW(ps, lr=LR, weight_decay=0.0)
            X = x[:t]
            for _ in range(cfg["steps"]):
                opt.zero_grad()
                z = net2(X)[:, :3]
                Xa = T7.strong_dev(X, gen) if cfg["views"] == "stock" else aug_pw(X, gen)
                za = net2(Xa)[:, :3]
                logp_all = torch.log_softmax(za, 1)
                kl = F.kl_div(logp_all, torch.softmax(z.detach(), 1), reduction="batchmean")
                if cfg["batch"] and t > cfg["batch"]:
                    idx = torch.randperm(t, generator=gen, device=dev)[:cfg["batch"]]
                else:
                    idx = torch.arange(t, device=dev)
                logp = logp_all[idx]
                p = logp.exp()
                pl = -(Q[idx] * logp).sum(1).mean()
                pm = p.mean(0)
                im = -(p * logp).sum(1).mean() + (pm * pm.clamp_min(1e-8).log()).sum()
                loss = pl + cfg["lam_im"] * im + LAM_KL * kl
                if cfg["anchor"]:
                    loss = loss + cfg["anchor"] * sum(((a - b) ** 2).sum() for a, b in zip(ps, theta0))
                loss.backward()
                opt.step()
                steps += 1
            hi = min(n, t + every)
            with torch.no_grad():
                out[t:hi] = torch.softmax(net2(x[:hi])[:, :3].double(), 1).cpu().numpy()[t:hi]
        return dict(base=base, r15=out.copy(), r15c250=out), steps

    return run_stream
