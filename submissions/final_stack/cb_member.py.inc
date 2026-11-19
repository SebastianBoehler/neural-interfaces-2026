class CBMember:
    """CBraMod-C slot member: causal prefix EA (5 prior steps), per-seed CBraMod encoder + task head, probability mean.

    Reproduces laneC2/cbmod.py::Member.forward with the PROTEUS montage head (task rows 0-2 only) and laneC/data.py::align
    for one stream (the session), one window per call."""

    K, SCALE, EMB = 5, 0.15, 200

    def __init__(self, blobs, dev):
        from braindecode.models import CBraMod

        self.dev, self.nets, self.heads, self.priors = dev, [], [], []
        for blob in blobs:
            s = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            net = CBraMod(n_outputs=4, n_chans=41, n_times=800, sfreq=200, return_encoder_output=True)
            net.load_state_dict({k: v.float() for k, v in s["net"].items()}, strict=True)
            head = torch.nn.Linear(41 * self.EMB, 3)
            head.load_state_dict({"weight": s["head_w"].float(), "bias": s["head_b"].float()}, strict=True)
            self.nets.append(net.to(dev).eval())
            self.heads.append(head.to(dev).eval())
            self.priors.append(s["prior"].double().numpy())
        if not self.nets or any(p.shape != (41, 41) or not np.isfinite(p).all() for p in self.priors):
            raise ValueError("invalid CBraMod member blobs")
        self.shared = all(np.array_equal(self.priors[0], p) for p in self.priors)
        self.reset()

    def reset(self):
        self.cov_sum = np.zeros((41, 41))
        self.n = 0

    def whiten(self, x, prior):
        s, u = np.linalg.eigh((self.cov_sum + self.K * prior) / (self.K + self.n))
        s = np.maximum(s, np.trace(self.cov_sum + self.K * prior) / (self.K + self.n) * 1e-5 / 41)
        return (((u * (s ** -0.5)[None, :]) @ u.T) @ x).astype(np.float32)

    @torch.no_grad()
    def prob(self, raw):
        if raw.shape != (41, 480):
            raise ValueError("CBraMod member needs the native 41x480 window")
        x = raw.astype(np.float64)
        x -= x.mean(-1, keepdims=True)
        self.cov_sum = self.cov_sum + x @ x.T / x.shape[1]
        self.n += 1
        aligned = None
        out = []
        for net, head, prior in zip(self.nets, self.heads, self.priors):
            if aligned is None or not self.shared:
                aligned = torch.from_numpy(self.whiten(x, prior)).to(self.dev)[None]
            z = torch.nn.functional.interpolate(aligned, size=800, mode="linear", align_corners=False)
            f = net(z.clamp(-15, 15) * self.SCALE, return_features=True)["features"]
            out.append(head(f.permute(0, 2, 1, 3).flatten(2)).mean(1).softmax(1)[0].double().cpu().numpy())
        q = np.mean(out, axis=0)
        if not np.isfinite(q).all() or abs(q.sum() - 1) > 1e-6:
            raise ValueError("invalid CBraMod probabilities")
        return q
