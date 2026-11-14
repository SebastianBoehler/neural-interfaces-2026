"""v6.2c: the run-level count filter on the emitted posterior (ledger s228c; dev form studies/14nov/rundetect:
detect.py 'lag', structured.py 'wprior', jointcount.py urn mixture). It reads only the window and the package's final
posterior and changes only the label that predict() emits: gates, shadow, routing and fine-tuning labels inside step()
never see it. State lives on the model and is reset with the session; numpy on the CPU, no clock, deterministic.

Per session, in delivery order:
- view: log-variance per channel in six bands (4th-order Butterworth, zero-phase), rounded to float32 as the dev cache,
  projected on 6 run-discriminant directions (W) and divided by their within-run SD (sw);
- detector: at window t = W in v6_cf_checks (W windows seen), 'lag' = mean distance of the view over pairs at most 3
  apart / pairs at least 15 apart among the first W windows; below v6_cf_thr[W] it latches for the rest of the session;
- filter: one urn per design hypothesis h (block lengths Ls[h], log prior lp[h], per-block cap; counts of each class
  clipped at K - 1) on posterior^tau; the output is the mixture sum_h w_h pred_h with w_h from the prior and the urns'
  predictive evidence of windows < t, normalised. It runs from window 0; the emitted label switches to it from the
  latch window on.
Any exception: counted, one stderr line, and the unfiltered posterior for the rest of the session."""
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import pdist
from scipy.special import logsumexp


class CountFilter:
    def __init__(self, z):
        self.W, self.sw = np.asarray(z["v6_cf_W"], np.float64), np.asarray(z["v6_cf_sw"], np.float64)
        fs = float(z["v6_cf_fs"])
        self.sos = [butter(4, tuple(b), "bandpass", fs=fs, output="sos") for b in np.asarray(z["v6_cf_bands"], np.float64)]
        self.checks = [int(w) for w in z["v6_cf_checks"]]
        self.thr = dict(zip(self.checks, np.asarray(z["v6_cf_thr"], np.float64)))
        Ls, self.lp, self.cap = np.asarray(z["v6_cf_Ls"]), np.asarray(z["v6_cf_lp"], np.float64), np.asarray(z["v6_cf_cap"])
        self.bounds = np.concatenate([np.zeros((len(Ls), 1), int), np.cumsum(Ls, 1), np.full((len(Ls), 1), 10 ** 6)], 1)
        self.tau, self.floor, K = float(z["v6_cf_tau"]), float(z["v6_cf_floor"]), int(z["v6_cf_K"])
        self.KK = np.stack(np.meshgrid(np.arange(K), np.arange(K), np.arange(K), indexing="ij"), -1)
        self.H, self.K = len(Ls), K
        self.reset_state()

    def reset_state(self):
        self.report()
        self.t, self.V, self.latch, self.dead = 0, [], None, False
        self.al = np.zeros((self.H, self.K, self.K, self.K))
        self.cum, self.blk = np.zeros(self.H), None
        self.log = dict(err=0, first="")

    def report(self):
        log = getattr(self, "log", None)
        if log is not None and self.t:
            print(f"[count-filter] windows {self.t} latched_at {self.latch} errors {log['err']}"
                  f"{' first: ' + log['first'] if log['first'] else ''}", file=sys.stderr, flush=True)

    def view(self, x):
        X = np.asarray(x, dtype=np.float64)
        F = np.concatenate([np.log(sosfiltfilt(sos, X, axis=-1).var(-1) + 1e-12) for sos in self.sos])
        F = F.astype(np.float32).astype(np.float64)  # the dev view was cached as float32 (common.blv)
        return (F @ self.W) / self.sw

    def lag(self, n):
        v = np.asarray(self.V[:n]); d = pdist(v); i, j = np.triu_indices(len(v), 1); g = j - i
        return d[g <= 3].mean() / d[g >= 15].mean()

    def urn(self, p):
        """One window of every hypothesis' urn (jointcount.urn's loop body) -> (H, 3) unnormalised predictive."""
        t = self.t
        blk = (self.bounds[:, 1:-1] <= t).sum(1)
        reset = np.ones(self.H, bool) if self.blk is None else blk != self.blk
        self.blk, cap = blk, self.cap[np.arange(self.H), blk]
        al = self.al
        al[reset] = 0; al[reset, 0, 0, 0] = 1.0
        L = np.clip(p, 1e-9, 1) ** self.tau; L /= L.sum()
        tr = np.maximum(cap[:, None, None, None, None] - self.KK[None], 0) + self.floor; tr /= tr.sum(-1, keepdims=True)
        v = al[..., None] * tr * L; out = v.reshape(self.H, -1, 3).sum(1); new = np.zeros_like(al)
        new[:, 1:] += v[:, :-1, :, :, 0]; new[:, -1] += v[:, -1, :, :, 0]
        new[:, :, 1:] += v[:, :, :-1, :, 1]; new[:, :, -1] += v[:, :, -1, :, 1]
        new[:, :, :, 1:] += v[:, :, :, :-1, 2]; new[:, :, :, -1] += v[:, :, :, -1, 2]
        self.al = new / out.sum(1)[:, None, None, None]
        return out

    def step(self, x, p):
        """x: the window as step() saw it; p: the package's posterior for it. -> the posterior predict() takes the
        label from (p itself before the latch, after an error, and for the rest of a session after an error)."""
        if self.dead:
            return p
        try:
            t = self.t
            if self.latch is None and t in self.thr and self.lag(t) < self.thr[t]:  # windows 0..t-1 seen
                self.latch = t
            self.V.append(self.view(x))
            pr = self.urn(np.asarray(p, np.float64))
            lw = self.lp + self.cum; w = np.exp(lw - logsumexp(lw))
            o = (w[:, None] * pr).sum(0); o = o / o.sum()
            self.cum = self.cum + np.log(np.maximum(pr.sum(1), 1e-300))
            self.t += 1
            if not np.isfinite(o).all():
                raise FloatingPointError("count filter output not finite")
            return o if self.latch is not None else p
        except Exception as e:  # the rest of the session emits the unfiltered posterior
            self.dead, self.t = True, self.t + 1
            try:
                self.log["err"] += 1
                self.log["first"] = self.log["first"] or repr(e)[:200]
                print(f"[count-filter] error at window {self.t - 1}: {repr(e)[:200]}; unfiltered for the rest of the "
                      "session", file=sys.stderr, flush=True)
            except Exception:
                pass
            return p
