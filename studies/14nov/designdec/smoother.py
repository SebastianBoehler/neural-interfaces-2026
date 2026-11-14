"""DESIGNDEC smoother (ledger s239): the run urn of runcount.py / jointcount.py with a prefix smoother.
Per session, H boundary hypotheses (block starts `reset` (T, H), caps `cap` (T, H), log prior `lp` (H,)). step(t, p)
is jointcount.urn's forward step for window t (bit-identical). soft(t) gives, for the windows < t:
  smooth  p(y_s | outputs of windows < t, urn) mixed over hypotheses with the weights of windows < t (forward-backward
          inside each hypothesis' current block; a finished block keeps its last smoothing, which saw the whole block)
  fwd     the forward-filtered posterior as the filter emitted it at s (weights of windows < s)
  lik     the window's own likelihood (output^TAU, normalised), for the extrinsic message smooth / lik."""
import os, sys
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../rundetect"))
import jointcount as J  # noqa: E402

K, KK = J.K, J.KK


def _tr(cap):  # cap (H,) -> (H, K, K, K, 3), jointcount.urn's transition
    tr = np.maximum(cap[:, None, None, None, None] - KK[None], 0) + J.FLOOR
    return tr / tr.sum(-1, keepdims=True)


def _shift(b):  # b (H, K, K, K) -> (H, K, K, K, 3): b at counts + e_y, clipped at K - 1 (the forward's clip)
    return np.stack([np.concatenate([b[:, 1:], b[:, -1:]], 1), np.concatenate([b[:, :, 1:], b[:, :, -1:]], 2),
                     np.concatenate([b[:, :, :, 1:], b[:, :, :, -1:]], 3)], -1)


class PrefixUrn:
    def __init__(self, lp, reset, cap):
        self.lp, self.reset, self.cap = np.asarray(lp, float), np.asarray(reset, bool), np.asarray(cap)
        T, H = self.reset.shape; assert self.reset[0].all()
        self.al = np.zeros((H, K, K, K)); self.pre = np.zeros((T, H, K, K, K), np.float32)
        self.L = np.zeros((T, 3)); self.pred = np.zeros((T, H, 3)); self.lev = np.zeros((T, H)); self.sm = np.zeros((T, H, 3))
        self.start = np.zeros((T, H), int)  # block start of window t under each hypothesis
        for t in range(T):
            self.start[t] = np.where(self.reset[t], t, self.start[t - 1] if t else 0)
        self.n = 0

    def step(self, t, p):
        assert t == self.n; self.n += 1; al = self.al
        al[self.reset[t]] = 0; al[self.reset[t], 0, 0, 0] = 1.0; self.pre[t] = al
        L = np.clip(p, 1e-9, 1) ** J.TAU; L /= L.sum(); self.L[t] = L
        v = al[..., None] * _tr(self.cap[t]) * L; out = v.reshape(len(al), -1, 3).sum(1); new = np.zeros_like(al)
        new[:, 1:] += v[:, :-1, :, :, 0]; new[:, -1] += v[:, -1, :, :, 0]
        new[:, :, 1:] += v[:, :, :-1, :, 1]; new[:, :, -1] += v[:, :, -1, :, 1]
        new[:, :, :, 1:] += v[:, :, :, :-1, 2]; new[:, :, :, -1] += v[:, :, :, -1, 2]
        self.al = new / out.sum(1)[:, None, None, None]; self.pred[t] = out; self.lev[t] = np.log(np.maximum(out.sum(1), 1e-300))
        if t + 1 < len(self.reset) and self.reset[t + 1].any():
            self.smooth(t + 1)  # a block ends at t: smooth it over its whole length now (a later soft() may not revisit it)

    def weights(self, t):  # hypothesis weights from windows < t (jointcount.filt_joint, mode urn)
        lw = self.lp + self.lev[:t].sum(0); return np.exp(lw - logsumexp(lw))

    def fwd(self, t):  # (t, 3): the filter's emitted posterior at each s < t
        cum = np.vstack([np.zeros(len(self.lp)), np.cumsum(self.lev[:t], 0)[:-1]]) + self.lp
        w = np.exp(cum - logsumexp(cum, 1, keepdims=True)); pr = self.pred[:t] / self.pred[:t].sum(2, keepdims=True)
        o = (w[..., None] * pr).sum(1); return o / o.sum(1, keepdims=True)

    def smooth(self, t):  # (t, 3)
        assert self.n >= t > 0; a = self.start[t - 1]; s0 = a.min(); H = len(self.lp); b = np.ones((H, K, K, K))
        for s in range(t - 1, s0 - 1, -1):
            lw = _tr(self.cap[s]) * self.L[s]; bs = _shift(b) * lw
            j = (self.pre[s][..., None] * bs).reshape(H, -1, 3).sum(1); live = s >= a
            self.sm[s, live] = j[live] / j[live].sum(1, keepdims=True)
            b = bs.sum(-1); b /= b.reshape(H, -1).max(1)[:, None, None, None]
        assert (self.sm[:t].sum(2) > 0).all(), "a window before t was never smoothed"
        o = (self.weights(t)[None, :, None] * self.sm[:t]).sum(1); return o / o.sum(1, keepdims=True)

    def soft(self, t, kind):
        if kind == "fwd":
            return self.fwd(t)
        sm = self.smooth(t)
        if kind == "rep":
            return sm
        m = sm / self.L[:t]; return m / m.sum(1, keepdims=True)  # ext: the urn's extrinsic message (caller multiplies it in)
