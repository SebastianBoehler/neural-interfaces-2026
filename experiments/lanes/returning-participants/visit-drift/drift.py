"""Lane D3 drift estimator (council_20nov/research2_codex.md idea 3). Plain numpy; build_drift.py pastes the class below the marker into
submission.py (which already has `np`), tests import it. MODE 0 off (bitwise no-op), 1 fixed half-shift (r = 0.5), 2 gated (corroboration + noise shrinkage).

Per window (after RouteB, only if a gate routes the window to person j whose enrolment row is finite): offset of the 12 first view PCs from
strictly earlier windows, u_i - sum_c p_i(c) mu_enrol[j,c], averaged in two interleaved halves (parity of the window index) from running
sums (O(1) per window); both halves need >= 20 windows. Gated: the two half offsets must have a positive dot product in enrolment
variance units, r = clip(1 - tr(V_noise)/||Delta||^2, 0, 0.5) on the pooled standardised offset. Centroids shift by r Delta; the output
logit of class c moves by the diagonal-Gaussian log-likelihood difference (shifted vs original centroid), clipped to +-0.3."""
# --- pasted from here ---
D3_K, D3_MIN, D3_CLIP = 12, 20, 0.3


class Drift:
    def __init__(self, z, mode):
        self.mode = int(mode)
        if self.mode:
            for k in ("d3_mu", "d3_var", "d3_n"):
                if k not in z:
                    raise ValueError(f"drift mode {mode} needs {k} in the package")
            self.mu, self.var, self.cnt = (np.asarray(z[k], dtype=np.float64) for k in ("d3_mu", "d3_var", "d3_n"))
            if not (self.mu.shape[1:] == (3, D3_K) and self.var.shape == self.mu.shape and len(self.mu) == len(z["route_names"])):
                raise ValueError("drift rows do not match the people of the package")
        self.reset_state()

    def reset_state(self):
        self.t = 0
        self.S = [dict(n=0, u=np.zeros(D3_K), p=np.zeros(3), uu=np.zeros(D3_K), up=np.zeros((D3_K, 3)), pp=np.zeros((3, 3))) for _ in (0, 1)]
        self.stats = dict(routed=0, nan_row=0, short=0, uncorroborated=0, active=0, r_sum=0.0, shift_sum=0.0, max_corr=0.0)

    def _add(self, u, p):
        s = self.S[self.t % 2]
        s["n"] += 1
        s["u"] += u
        s["p"] += p
        s["uu"] += u * u
        s["up"] += np.outer(u, p)
        s["pp"] += np.outer(p, p)
        self.t += 1

    def correction(self, u, j):
        """(per-class logit change, r, shift) for the window with view u[:12] routed to person j, or None."""
        mu, var = self.mu[j], self.var[j]
        st = self.stats
        if not (np.isfinite(mu).all() and np.isfinite(var).all()):
            st["nan_row"] += 1
            return None
        n = np.array([s["n"] for s in self.S])
        if n.min() < D3_MIN:
            st["short"] += 1
            return None
        w = np.maximum(self.cnt[j], 1.0)
        v = np.maximum(var, 1e-8)
        pooled = np.maximum((w[:, None] * var).sum(0) / w.sum(), 1e-8)  # enrolment within-class variance, pooled over classes
        offs = [(s["u"] - s["p"] @ mu) / s["n"] for s in self.S]
        delta = (n[0] * offs[0] + n[1] * offs[1]) / n.sum()
        if self.mode == 2:
            if float((offs[0] * offs[1] / pooled).sum()) <= 0:
                st["uncorroborated"] += 1
                return None
            ssq = sum(s["uu"] - 2 * (mu.T * s["up"]).sum(1) + np.einsum("cd,ce,ed->d", mu, s["pp"], mu) for s in self.S) / n.sum()
            vres = np.maximum(ssq - delta ** 2, 0.0)  # residual variance of u - p mu about the common offset (includes class overlap: conservative)
            pbar = sum(s["p"] for s in self.S) / n.sum()
            noise = float((vres / pooled).sum() / n.sum() + (pbar[:, None] ** 2 * var / w[:, None] / pooled).sum())
            r = float(np.clip(1.0 - noise / max(float((delta ** 2 / pooled).sum()), 1e-12), 0.0, 0.5))
        else:
            r = 0.5
        shift = r * delta
        d = u[None, :] - mu
        ll = (shift[None, :] * d / v - 0.5 * shift[None, :] ** 2 / v).sum(1)
        return np.clip(ll, -D3_CLIP, D3_CLIP), r, shift

    def apply(self, out, u, shadow, gc, gi):
        """out: RouteB's posterior of this window; u: its 256-PC view; shadow: unrouted posterior; gc, gi: gate decisions (j, _, ramp)."""
        if not self.mode:
            return out
        u12, p = np.asarray(u[:D3_K], dtype=np.float64), np.asarray(shadow, dtype=np.float64)
        j, r = -1, 0.0
        for g in (gc, gi):
            if g is not None and g[2] > r:
                j, r = int(g[0]), float(g[2])
        res = None
        if r > 0 and j >= 0:
            self.stats["routed"] += 1
            res = self.correction(u12, j)
        self._add(u12, p)  # after the estimate: strictly earlier windows only
        if res is None:
            return out
        ll, rr, shift = res
        if rr <= 0:  # gated arm: the offset is not trusted, nothing moves
            return out
        o = np.log(np.clip(np.asarray(out, dtype=np.float64), 1e-12, None)) + ll
        o = np.exp(o - o.max())
        o /= o.sum()
        if not np.isfinite(o).all():
            return out
        st = self.stats
        st["active"] += 1
        st["r_sum"] += rr
        st["shift_sum"] += float(np.abs(shift).sum())
        st["max_corr"] = max(st["max_corr"], float(np.abs(ll).max()))
        return o
