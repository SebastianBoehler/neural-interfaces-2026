class TeamBlendStack(StackEnsemble):
    TEAM_B_MIX = __BLEND__
    TEAM_B_MEMBERS = __MEMBERS__
    TEAM_B_SHA = __SHA__

    def __init__(self, z, dev, root):
        self.team_b = None
        super().__init__(z, dev, root)
        if self.TEAM_B_MIX:
            blob = (Path(root) / "team_b.pt").read_bytes()
            if hashlib.sha256(blob).hexdigest() != self.TEAM_B_SHA:
                raise ValueError("Team B weights differ from the signed bank")
            pack = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            self.team_b = _TEAM_B["Encoder"](pack, dev, self.TEAM_B_MEMBERS)

    def reset_state(self):
        super().reset_state()
        if self.team_b is not None:
            self.team_b.reset()

    def mix6(self, mix, members, p_ft):
        raw = self.hard_window
        m = super().mix6(mix, members, p_ft)
        if self.team_b is None or not members:
            return m
        q = self.team_b.posterior(torch.from_numpy(raw)).cpu().numpy()
        if q.shape != (3,) or not np.isfinite(q).all() or (q < 0).any() or abs(float(q.sum()) - 1) > 1e-6:
            raise ValueError("invalid Team B posterior")
        return (1 - self.TEAM_B_MIX) * m + self.TEAM_B_MIX * q
