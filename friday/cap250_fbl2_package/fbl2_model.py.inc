class FBLightEnsemble(Ensemble6):
    """Double the live FBLight family only in the final routed ensemble."""

    def __init__(self, z, device, root):
        super().__init__(z, device, root)
        members = tuple(self.ft.slots["fbl"])
        if (self.routed != 1 or self.v6r or not self.r15 or members != tuple(range(59, 65))
                or self.ft.member["fbl"] != members[0]):
            raise ValueError("FBLight doubling requires the frozen cap250 layout")
        weights = self.w[1 + np.asarray(members)]
        mass = float(weights.sum())
        post_rest = (1 - self.w_rest) * mass
        if (self.w_rest != 1 / 16 or abs(post_rest - 1 / 16) > 1e-16 or
                abs(self.routeB.a - 1 / 16) > 1e-16 or
                not np.allclose(weights, 1 / 90, atol=1e-16, rtol=0)):
            raise ValueError("unexpected frozen cap250 family or route weight")
        self.fbl_members = members
        self.fbl_weights = weights
        self.fbl_mass = mass
        self.fbl_post_rest = post_rest
        self.routeB.a = 1 / 17

    def mix6(self, mix, members, p_ft):
        native = super().mix6(mix, members, p_ft)
        if not members:
            return native
        frozen = np.stack([members[i] for i in self.fbl_members])
        adapted = (self.fbl_weights[:, None] * frozen).sum(0) / self.fbl_mass
        if "fbl" in p_ft:
            shifted = np.clip(adapted + p_ft["fbl"] - frozen[0], 1e-6, None)
            adapted = shifted / shifted.sum()
        return (native + self.fbl_post_rest * adapted) / (1 + self.fbl_post_rest)
