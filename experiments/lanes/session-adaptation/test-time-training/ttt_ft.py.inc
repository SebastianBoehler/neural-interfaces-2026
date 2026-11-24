class TTTFineTune(FineTune):
    """Lane T: FineTune with team C's recipe in the refit. Same schedule (t0, every, cap, guards), same warm start, same Adam restart
    per refit. Targets: the soft posteriors (the relabelled prefix when r15 gives it, else the package's own earlier outputs),
    Sinkhorn-balanced; loss = soft CE on the strong view + IM_LAM * information maximisation + KL(clean || strong) * lam
    (+ ANCHOR * ||theta - theta_source||^2 for the heavy form, with the loss on at most BATCH random prefix windows per step;
    the forward always runs over the whole prefix because StreamBN is causal in window order)."""
    FORM = "light"  # set per package: "light" | "heavy"
    FORMS = {"light": dict(steps=0, batch=0, anchor=0.0, views="stock"), "heavy": dict(steps=20, batch=64, anchor=1e-2, views="pw"),
             "lite5": dict(steps=5, batch=64, anchor=1e-2, views="pw"),
             "sparse20": dict(steps=20, batch=64, anchor=1e-2, views="pw", every=40)}
    IM_LAM = 0.3

    def __init__(self, z, w, dev, root):
        super().__init__(z, w, dev, root)
        if self.FORMS[self.FORM].get("every"):  # sparse20: refits at 20, 60, ... (the relabelled-target grid is 10 wide, 40 is a multiple)
            assert self.FORMS[self.FORM]["every"] % self.every == 0
            self.every = self.FORMS[self.FORM]["every"]

    def reset_state(self):
        super().reset_state()
        self.post = []

    def relabel(self, outs):
        super().relabel(outs)
        self.post = [np.asarray(o, dtype=np.float64) for o in outs]

    def step(self, members, inputs, out, relab_fn=None):
        res = super().step(members, inputs, out, relab_fn)
        self.post.append(np.asarray(out, dtype=np.float64))
        assert len(self.post) == len(self.lab)
        return res

    @property
    def STEPS_PER_FAMILY(self):
        return self.FORMS[self.FORM]["steps"] or self.n_steps

    def refit(self, t, relab_fn=None):
        """the base refit (relabelling, guards, unoptimised executor) with the soft relabelled prefix captured for _refit"""
        cap = self._cap = {}

        def capture(tt):
            cap["P"] = relab_fn(tt) if relab_fn is not None else None
            return cap["P"]
        super().refit(t, capture)

    def _refit(self, t, lab, wt):
        P = self._cap.get("P")
        if P is None:
            P = np.stack(self.post[:t])
        assert np.asarray(P).shape == (t, 3)
        self._ttt(t, np.asarray(P, dtype=np.float32))

    @staticmethod
    def _sinkhorn(P, iters=10):
        Q = P.clamp_min(1e-8)
        for _ in range(iters):
            Q = Q / Q.sum(0, keepdim=True)
            Q = Q / Q.sum(1, keepdim=True)
        return Q

    @staticmethod
    def _view(X, g, shift=15, scale=0.15, drop=0.1, noise=0.2):
        B, C, T = X.shape
        s = torch.randint(-shift, shift + 1, (B, 1, 1), generator=g, device=X.device)
        Xs = torch.gather(X, 2, ((torch.arange(T, device=X.device)[None, None] + s) % T).expand(B, C, T))
        Xs = Xs * torch.exp(scale * torch.randn(B, 1, 1, generator=g, device=X.device))
        Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g, device=X.device)
        return Xs * (torch.rand(B, C, 1, generator=g, device=X.device) > drop).float()

    def _ttt(self, t, P):
        f = dict(self.FORMS[self.FORM])
        steps = f["steps"] or self.n_steps
        Q = self._sinkhorn(torch.from_numpy(P).to(self.dev))
        for s, m in self.nets.items():
            X = torch.from_numpy(np.stack(self.X[s][:t])).to(self.dev)
            ps = self.params(m)
            for p in ps:
                p.requires_grad_(True)
            opt = torch.optim.Adam(ps, lr=self.lr)
            for _ in range(steps):
                opt.zero_grad()
                self.zero(m)
                with torch.no_grad():
                    zc = m(X)[:, :3]
                self.zero(m)
                Xa = strong(X, self.gens[s]) if f["views"] == "stock" else self._view(X, self.gens[s])
                logp_all = F.log_softmax(m(Xa)[:, :3], 1)
                idx = torch.randperm(t, generator=self.gens[s], device=self.dev)[:f["batch"]] if f["batch"] and t > f["batch"] \
                    else torch.arange(t, device=self.dev)
                logp = logp_all[idx]
                p = logp.exp()
                pm = p.mean(0)
                im = -(p * logp).sum(1).mean() + (pm * pm.clamp_min(1e-8).log()).sum()
                loss = -(Q[idx] * logp).sum(1).mean() + self.IM_LAM * im \
                    + self.lam * F.kl_div(logp_all, F.softmax(zc, 1), reduction="batchmean")
                if f["anchor"]:
                    loss = loss + f["anchor"] * sum(((a - b) ** 2).sum() for a, b in zip(ps, self.theta0[s]))
                loss.backward()
                opt.step()
            self.steps += steps
            with torch.no_grad():
                self.zero(m)
                m(X)
                self.dtheta = max(self.dtheta, max(float((p - q).abs().max()) for p, q in zip(self.params(m), self.theta0[s])))
        self.refits += 1
