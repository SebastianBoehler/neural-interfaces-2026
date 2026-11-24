"""Lane I: eval-time moment rules for experiments/session_bn.SessionBN (S1 age-only prior count, S2 risk-weighted).

S0 = the stock eval (k=20): never touched. ShrinkSBN only changes the eval branch when mode == "s2"; every other mode calls
SessionBN.forward unchanged, so S0 / S1 (k set through SessionNet.set_k) are the stock code path."""
import numpy as np
import torch

from session_bn import SessionBN

K0 = 4.0  # pseudo-windows of the source within-session variance in the online W estimate


class ShrinkSBN(SessionBN):
    mode = "stock"
    T = Wsrc = None

    def forward(self, x):
        if self.training or self.mode != "s2":
            return super().forward(x)
        if self.T is None or self.Wsrc is None:
            raise RuntimeError("S2 needs source moments (calibrate first)")
        red = tuple(range(2, x.dim()))
        r1, r2 = x.mean(red), (x * x).mean(red)
        n = torch.arange(1, len(x) + 1, device=x.device, dtype=x.dtype)[:, None]
        m1, m2 = r1.cumsum(0) / n, r2.cumsum(0) / n
        a = self.weight_s2(n, r1, m1)
        rm, rv = self.running_mean, self.running_var
        mean = rm + a * (m1 - rm)
        var = ((rv + rm ** 2) + a * (m2 - (rv + rm ** 2)) - mean ** 2).clamp_min(0)
        shape = (len(x), -1) + (1,) * len(red)
        out = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return self._affine(out, (1, -1) + (1,) * len(red))

    def weight_s2(self, n, r1, m1):
        """a_{c,t} = n T / (n T + W_t); W_t from delivered windows' moments only (class-blind)."""
        ss = (r1 * r1).cumsum(0) - n * m1 * m1
        W = (K0 * self.Wsrc + ss.clamp_min(0)) / (K0 + n)
        return n * self.T / (n * self.T + W)


def install(net):
    """Swap every SessionBN of a SessionNet to ShrinkSBN (state untouched); returns the modules."""
    mods = net.sbns()
    assert mods, "no SessionBN in net"
    for m in mods:
        m.__class__ = ShrinkSBN
    return mods


@torch.no_grad()
def calibrate(net, d, train, mods):
    """T_c, Wsrc_c from the TRAINING people's sessions only: per session the mean and variance across windows of the
    per-window channel moment r1 at each SessionBN input. Between-session variance of session means minus its sampling part."""
    rec = {id(m): [] for m in mods}
    hooks = [m.register_forward_hook(lambda mod, inp, out: rec[id(mod)].append(inp[0].mean(tuple(range(2, inp[0].dim()))).double().cpu()))
             for m in mods]
    net.net.eval()
    try:
        st = d["stream"][train]
        for s in dict.fromkeys(st):
            net.predict_proba_stream(train[st == s])
    finally:
        for h in hooks:
            h.remove()
    sess = len(dict.fromkeys(d["stream"][train]))
    for m in mods:
        chunks = rec[id(m)]
        assert len(chunks) == sess, (len(chunks), sess)
        means = torch.stack([c.mean(0) for c in chunks])
        wv = torch.stack([c.var(0, unbiased=True) for c in chunks])
        ns = torch.tensor([len(c) for c in chunks], dtype=torch.float64)[:, None]
        W = wv.mean(0)
        T = (means.var(0, unbiased=True) - (wv / ns).mean(0)).clamp_min(1e-3 * W)  # positive part, floored
        m.Wsrc, m.T = W.to(m.running_mean).clone(), T.to(m.running_mean).clone()
    return {"T_over_W_median": float(np.median([float((m.T / m.Wsrc).median()) for m in mods]))}
