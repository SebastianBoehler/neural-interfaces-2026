"""Trained-in per-session latent normalisation (5 Nov; BEETL 2021 winner trick, Bakas et al.
arXiv 2202.03267; domain-specific BN as in TSMNet, Kobler et al. 2022).

Every batch-norm layer of a compact net on EA input uses *per-session* statistics while training and
causal running statistics of the test session at test time. Copied from the dev harness
(scratchpad t5/latent_norm/lnorm.py, md5 c3ddd81e, which produced the fb_sbn_rot05 dev result; lnorm_r15.py
for the R15 members' twins, make("NAME_sbn")) plus the stateful TorchScript export (StreamBN, prototype
stream_export.py).

SessionBN (drop-in for a BatchNorm layer):
  train: the batch is G chunks of m rows, each chunk from one session (SessionNet's sampler); each chunk is
         normalised with its own mean/var over (rows, non-channel dims). Running stats are EMAs of the chunk
         means and within-chunk variances (the population prior: a typical session).
  eval:  rows are one session in stream order (protocol.predict_streams); window t is normalised with the
         expanding stats of windows 0..t blended with the running stats as k pseudo-windows (k=None: running
         stats only). Label-free and causal; state resets per session.
StreamBN: the same eval formula for one window per call; buffers s1, s2 (sums of per-window moments) and n
carry the session's statistics across calls (submission.py SJit zeroes them per session). to_stream(net, k)
converts a trained net for tracing.
"""
import copy
import re

import numpy as np
import torch
from torch import nn

import dg
from deep import AlignedNet, augment


class SessionBN(nn.Module):
    def __init__(self, C, eps=1e-5, momentum=0.1, affine=True, causal=False, k_train=20.0):
        super().__init__()
        self.C, self.eps, self.momentum, self.causal, self.k_train = C, eps, momentum, causal, k_train
        self.weight = nn.Parameter(torch.ones(C)) if affine else None
        self.bias = nn.Parameter(torch.zeros(C)) if affine else None
        self.register_buffer("running_mean", torch.zeros(C))
        self.register_buffer("running_var", torch.ones(C))
        self.m, self.k = 0, 20.0  # chunk size (train) and test prior pseudo-windows (eval), set by SessionNet

    def _affine(self, out, shape):
        if self.weight is None:
            return out
        return out * self.weight.view(shape) + self.bias.view(shape)

    def forward(self, x):
        red = tuple(range(2, x.dim()))
        shape = (1, -1) + (1,) * len(red)
        if self.training:
            m = self.m or len(x)
            G = len(x) // m
            assert G * m == len(x), (len(x), m)
            z = x.reshape(G, m, *x.shape[1:])
            zr = tuple(a + 1 for a in red)
            r1 = z.mean(zr) if red else z  # (G, m, C) per-row moments
            r2 = (z * z).mean(zr) if red else z * z
            if self.causal:
                rm, rv = self.running_mean.detach(), self.running_var.detach()
                k = self.k_train
                n = torch.arange(1, m + 1, device=x.device, dtype=x.dtype)[None, :, None]
                mean = (k * rm + r1.cumsum(1)) / (k + n)
                var = ((k * (rv + rm ** 2) + r2.cumsum(1)) / (k + n) - mean ** 2).clamp_min(0)
            else:
                mean = r1.mean(1, keepdim=True)
                var = (r2.mean(1, keepdim=True) - mean ** 2).clamp_min(0)
                mean, var = mean.expand(G, m, -1), var.expand(G, m, -1)
            with torch.no_grad():
                gm = r1.mean(1)
                gv = (r2.mean(1) - gm ** 2).clamp_min(0)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * gm.mean(0))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * gv.mean(0))
            mean, var = mean.reshape(len(x), -1), var.reshape(len(x), -1)
        else:
            rm, rv = self.running_mean, self.running_var
            if self.k is None:
                mean, var = rm.expand(len(x), -1), rv.expand(len(x), -1)
            else:
                r1 = x.mean(red) if red else x
                r2 = (x * x).mean(red) if red else x * x
                n = torch.arange(1, len(x) + 1, device=x.device, dtype=x.dtype)[:, None]
                mean = (self.k * rm + r1.cumsum(0)) / (self.k + n)
                var = ((self.k * (rv + rm ** 2) + r2.cumsum(0)) / (self.k + n) - mean ** 2).clamp_min(0)
        bshape = (len(x), -1) + (1,) * len(red)
        out = (x - mean.view(bshape)) / torch.sqrt(var.view(bshape) + self.eps)
        return self._affine(out, shape)


def swap_bn(net, causal, k_train):
    todo = [(p, n, c) for p in net.modules() for n, c in p.named_children()
            if isinstance(c, nn.modules.batchnorm._BatchNorm)]
    for parent, name, child in todo:
        sbn = SessionBN(child.num_features, child.eps, child.momentum or 0.1, child.affine, causal, k_train)
        with torch.no_grad():  # keep a pretrained trunk's BN (fresh nets: ones/zeros, as before)
            if child.affine:
                sbn.weight.copy_(child.weight), sbn.bias.copy_(child.bias)
            if child.running_mean is not None:
                sbn.running_mean.copy_(child.running_mean), sbn.running_var.copy_(child.running_var)
        setattr(parent, name, sbn.to(child.weight.device if child.affine else "cpu"))
    return len(todo)


class SessionNet(AlignedNet):
    """AlignedNet whose batches are chunks of ``group`` same-session rows and whose BN layers are SessionBN."""

    def __init__(self, d, group=32, causal=False, k_train=20.0, **kw):
        super().__init__(d, **kw)
        self.group, self.causal, self.k_train = group, causal, k_train

    def _fit(self, idx):
        st = self.d["stream"][idx]
        self.fit_subjects = set(np.unique(self.d["subject"][idx]).tolist())  # s279 guard: the people the net (and any donor) comes from
        self._row_sess = np.concatenate([np.full(int((st == s).sum()), j) for j, s in enumerate(dict.fromkeys(st))])
        super()._fit(idx)

    def _prepare(self):
        if getattr(self.net, "_sbn_done", False):
            return
        self.n_sbn = swap_bn(self.net, self.causal, self.k_train)
        self.net.to(self.dev)
        self.net._sbn_done = True
        for mod in self.sbns():
            mod.m = self.group

    def sbns(self):
        return [m for m in self.net.modules() if isinstance(m, SessionBN)]

    def set_k(self, k):
        for mod in self.sbns():
            mod.k = k

    def _batches(self, rows, rng):
        m, G = self.group, max(1, self.batch // self.group)
        chunks = []
        sess = self._row_sess[rows]
        for s in np.unique(sess):
            r = rows[sess == s]
            r = r[rng.permutation(len(r))]
            pad = (-len(r)) % m
            if pad:
                r = np.concatenate([r, rng.choice(r, pad, replace=pad > len(r))])
            chunks += list(r.reshape(-1, m))
        order = rng.permutation(len(chunks))
        chunks = [chunks[i] for i in order]
        return [np.concatenate(chunks[b:b + G]) for b in range(0, len(chunks), G)]

    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted, head_only=False, avg=None):
        assert not (head_only or avg is not None or weighted or self.dro or self.pbal or self.mstyle or self.adv)
        # 12 Nov: cmix and fsc were accepted here and never applied (ledger s51). Fail loudly instead of training a
        # net that silently lacks the requested augmentation; chunk-wise versions live in studies/12nov/netaug.py.
        assert not (self.cmix or self.fsc), "SessionNet does not apply cmix / fsc; use studies/12nov/netaug.VNet"
        assert not (self.cotrain or self.adabn), "SessionNet applies neither the co-training loss nor AdaBN (audit, ledger s52)"
        self._prepare()
        if self.am:  # s279: donors are rows of other people among the TRAINING rows of this fit (X is the fold's training set only)
            amg = torch.Generator().manual_seed(self.seed + 7919); sub = self._grp.cpu()
            self.am_stats = dict(mixed=0, batches=0, absdiff=0.0, bad_donor=0)
        dgs = self.dg_stats = dict(steps=0, coral_pairs=0, coral_pen=0.0, adv_loss=0.0, adv_on=0, dist_steps=0, dist_kl=0.0, snaps=0, snap_diff=0.0, avg_diff=0.0)
        if self.dann or self.coral:  # s285: person labels = self._grp, built from the fit's training rows only
            assert int(self._grp.max()) + 1 == len(self.fit_subjects), "person labels are not exactly the training people"
        if self.dist:  # s283: teacher posteriors of the training windows (global index -> row), tempered p^(1/T) renormalised; every training row must be covered
            assert self.teacher is not None, "dist without a teacher"
            tix, tpr = self.teacher; o = np.argsort(tix); pos = np.searchsorted(tix[o], self._order); assert (pos < len(tix)).all() and (tix[o][np.minimum(pos, len(tix) - 1)] == self._order).all(), "teacher misses training rows"
            tp = torch.from_numpy(np.clip(tpr[o][pos], 1e-8, 1.0) ** (1.0 / self.dist_T)).float(); self._tp = (tp / tp.sum(1, keepdim=True)).to(self.dev)
        tap = dg.Tap(self.net) if self.coral else None
        adv = dg.Adversary(self.net, len(self.fit_subjects), 0.0, lr, self.dev) if self.dann else None; snaps = []
        from deep import param_groups
        groups = param_groups(self.net, lr, self.lld, head_only)
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.wd)
        nb = len(self._batches(rows, np.random.default_rng(0)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=epochs * nb)
        for ep in range(epochs):
            for j in self._batches(rows, rng):
                xb, yb = X[j].to(self.dev), y[j].to(self.dev)
                if self.aug:
                    xb = augment(xb, gen)
                if self.am:
                    jt = torch.from_numpy(j); r = torch.randint(0, len(X), (len(j),), generator=amg)
                    for _ in range(50):  # redraw donors that belong to the same person
                        same = sub[r] == sub[jt]
                        if not same.any():
                            break
                        r[same] = torch.randint(0, len(X), (int(same.sum()),), generator=amg)
                    assert not (sub[r] == sub[jt]).any(), "am donor from the same person"  # r indexes X = the training rows only
                    xm, sel = dg.amp_mix(xb, X[r].to(self.dev), amg, self.am); dd = float((xm - xb).abs().sum())
                    assert dd > 0 or not sel.any(), "am changed nothing in a batch with selected windows"
                    self.am_stats["mixed"] += int(sel.sum()); self.am_stats["batches"] += 1; self.am_stats["absdiff"] += dd; xb = xm
                if self.rot:
                    xb = dg.rotate(xb, gen, self.rot)
                opt.zero_grad()
                zb = self.net(xb)[:, :3]; loss = loss_fn(zb, yb); dgs["steps"] += 1; jd = torch.from_numpy(j).to(self.dev)
                if self.dist:  # s283: CE + weight * T^2 * KL(teacher_T || student_T)
                    kl = torch.nn.functional.kl_div(torch.log_softmax(zb / self.dist_T, 1), self._tp[jd], reduction="batchmean") * self.dist_T ** 2
                    loss = loss + self.dist * kl; dgs["dist_steps"] += 1; dgs["dist_kl"] += float(kl)
                if tap:  # s285 CORAL: weight * penalty between per-person feature covariances in this batch
                    pen, npairs = dg.coral(tap.feat, self._grp[jd]); loss = loss + self.coral * pen; dgs["coral_pairs"] += npairs; dgs["coral_pen"] += float(pen)
                if adv:  # s285 DANN: person head behind gradient reversal, lambda ramp 0 -> dann (DANN schedule over training progress)
                    adv.lam = self.dann * (2 / (1 + np.exp(-10 * dgs["steps"] / (epochs * nb))) - 1); al = adv.loss(self._grp[jd]); loss = loss + al
                    dgs["adv_loss"] += float(al); dgs["adv_on"] += adv.lam > 0
                loss.backward()
                opt.step()
                if adv:
                    adv.step()
                sched.step()
            if self.swad and ep >= epochs // 2:  # s285 SWAD: snapshots of the last half of the epochs
                snaps.append({k: v.detach().clone() for k, v in self.net.state_dict().items()})
        for h in (tap, adv):
            if h:
                h.remove()
        if self.swad:
            assert len(snaps) >= 2, "SWAD needs at least two snapshots"; dgs["snaps"] = len(snaps)
            dgs["snap_diff"] = max(float((snaps[0][k] - snaps[-1][k]).abs().max()) for k in snaps[0] if snaps[0][k].is_floating_point())
            avg = dg.swad_average(snaps); dgs["avg_diff"] = max(float((avg[k] - snaps[-1][k]).abs().max()) for k in avg if avg[k].is_floating_point())
            self.net.load_state_dict(avg)


ARCH = {"sh": "shallow", "nx": "eegnex", "fb": "fbcnet"}
CFGS = {"plain": None, "sbn": dict(group=32), "sbn16": dict(group=16)}  # None: the plain AlignedNet twin


def make(cfg, d, seed):
    """CFG ARCH_VARIANT[_rot05], e.g. fb_sbn_rot05 (the dev harness's make()), or NAME_sbn: the R15 member
    NAME built exactly as run_cv builds it (MODS such as rot05, pretrained trunks such as ptm7r, seed) with its
    class switched to SessionNet (dev harness lnorm_r15.py md5 2cfff20e, e.g. fblight_ea_rot05_sbn)."""
    if cfg.endswith("_sbn"):
        import run_cv
        run_cv.SEED[0] = seed
        net = run_cv.resolve(cfg[:-4])(d)
        assert type(net) is AlignedNet, type(net)
        net.__class__ = SessionNet
        net.group, net.causal, net.k_train = 32, False, 20.0
        return net
    mdg = re.search(r"_(coral|dann|swad)$", cfg)  # s285: fb_sbn_rot05_coral / _dann / _swad (CORAL weight 1, DANN lambda 0.3, SWAD last 50 % of epochs; the arm is in the family name)
    dgk = mdg.group(1) if mdg else None; cfg = cfg[:mdg.start()] if mdg else cfg
    mam = re.search(r"_am(\d*)$", cfg)  # s279: fb_sbn_rot05_am = p 0.5; _am25 / _am100 = p 0.25 / 1.0 (the p is in the family name, hence in the output file name)
    am = (int(mam.group(1)) / 100 if mam.group(1) else 0.5) if mam else 0.0
    cfg = cfg[:mam.start()] if mam else cfg
    arch, _, var = cfg.partition("_")
    var, rot = (var[:-6], 0.05) if var.endswith("_rot05") else (var, 0.0)
    kw = CFGS[var]
    net = AlignedNet(d, arch=ARCH[arch], seed=seed) if kw is None else SessionNet(d, arch=ARCH[arch], seed=seed, **kw)
    net.rot, net.am = rot, am
    if dgk:
        setattr(net, dgk, {"coral": 1.0, "dann": 0.3, "swad": 1.0}[dgk])
    return net


class StreamBN(nn.Module):
    """SessionBN's eval formula for streamed windows, device-free for tracing (no baked device constants)."""

    def __init__(self, sbn: SessionBN, k: float):
        super().__init__()
        C = sbn.C
        self.k, self.eps = float(k), float(sbn.eps)
        self.register_buffer("rm", sbn.running_mean.detach().clone())
        self.register_buffer("rv", sbn.running_var.detach().clone())
        self.register_buffer("w", sbn.weight.detach().clone() if sbn.weight is not None else torch.ones(C))
        self.register_buffer("b", sbn.bias.detach().clone() if sbn.bias is not None else torch.zeros(C))
        self.register_buffer("s1", torch.zeros(C))
        self.register_buffer("s2", torch.zeros(C))
        self.register_buffer("n", torch.zeros(1))

    def forward(self, x):
        dims = [d for d in range(2, x.dim())]
        r1 = x.mean(dims) if len(dims) > 0 else x
        r2 = (x * x).mean(dims) if len(dims) > 0 else x * x
        c1, c2 = self.s1 + r1.cumsum(0), self.s2 + r2.cumsum(0)
        n = self.n + torch.ones_like(r1[:, :1]).cumsum(0)
        mean = (self.k * self.rm + c1) / (self.k + n)
        var = ((self.k * (self.rv + self.rm ** 2) + c2) / (self.k + n) - mean ** 2).clamp_min(0.0)
        self.s1.copy_(c1[-1])
        self.s2.copy_(c2[-1])
        self.n.copy_(n[-1])
        sd = torch.sqrt(var + self.eps)
        w, b = self.w, self.b
        for _ in dims:  # (B, C) -> (B, C, 1, ...)
            mean, sd, w, b = mean.unsqueeze(-1), sd.unsqueeze(-1), w.unsqueeze(-1), b.unsqueeze(-1)
        return (x - mean) / sd * w + b


STATE = ("s1", "s2", "n")  # buffer names that submission.py SJit.reset_state zeroes


def reset_stream(module):
    for name, b in module.named_buffers():
        if name.rsplit(".", 1)[-1] in STATE:
            b.zero_()


def to_stream(net, k):
    """Copy of a trained SessionNet's ``net`` (CPU, eval) with every SessionBN replaced by StreamBN(k)."""
    net = copy.deepcopy(net).cpu().eval()
    # metadata only (forward unchanged): unset braindecode properties break the trace's hasattr scan
    net._chs_info = net._chs_info or [{"ch_name": f"ch{c}"} for c in range(41)]
    net._sfreq = net._sfreq or 120.0
    net._input_window_seconds = net._input_window_seconds or 4.0
    todo = [(p, n, c) for p in net.modules() for n, c in p.named_children() if isinstance(c, SessionBN)]
    for p, n, c in todo:
        setattr(p, n, StreamBN(c, k))
    assert todo and not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in net.modules())
    return net
