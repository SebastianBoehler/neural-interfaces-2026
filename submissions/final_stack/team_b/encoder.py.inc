class Encoder:
    """The R1 enc arm: causal EA, prefix feature normalization, geometric mean of class posteriors."""

    def __init__(self, pack, device, members):
        if pack.get("cfg") or pack.get("experts") or pack.get("ext"):
            raise ValueError("expected a plain Team B R1 weight bank")
        roster = tuple(kind for kind, _ in pack["nets"])
        if roster != ("shallow",) * members + ("slow",) * members + ("fbc",) * members:
            raise ValueError(f"unexpected Team B member roster: {roster}")
        self.device = device
        self.nets = []
        for kind, state in pack["nets"]:
            net = Net(kind)
            net.load_state_dict(state)
            self.nets.append(net.to(device).eval())
        self.mask = fmask(1.0, 45.0).to(device)
        self.reset()

    def reset(self):
        self.windows, self.cov, self.embeddings = [], None, None

    @torch.no_grad()
    def posterior(self, window):
        x = detrend(window.to(self.device, torch.float32))
        self.windows.append(x)
        n = len(self.windows)
        xb = bandpass(x[None], self.mask)[0].double()
        c = xb @ xb.T / T
        self.cov = c if self.cov is None else self.cov + c
        W = whitener(self.cov, n).float()
        if n <= 16 or n % DEFAULTS["K"] == 0:
            xs = W @ torch.stack(self.windows)
            self.embeddings = [
                torch.cat([net.feats(xs[i:i + 64]) for i in range(0, n, 64)]) for net in self.nets
            ]
        else:
            xa = (W @ x)[None]
            self.embeddings = [
                torch.cat([emb, net.feats(xa)]) for net, emb in zip(self.nets, self.embeddings, strict=True)
            ]
        # Keep R1's full-prefix head evaluation and arithmetic for encoder parity.
        k = DEFAULTS["sfn_k"]
        logp = sum(torch.log_softmax(net.head(sfn(emb, k)), dim=1)
                   for net, emb in zip(self.nets, self.embeddings, strict=True)) / len(self.nets)
        return torch.softmax(logp, dim=1)[-1]
