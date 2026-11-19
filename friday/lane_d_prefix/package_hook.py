# This source is appended only to a hash-pinned copy of cap250.
LANE_D_ARM = "off"


class LaneDEnsemble(Ensemble6):
    def reset_state(self):
        super().reset_state()
        if LANE_D_ARM == "output":
            self.lane_d = PrefixBalance()

    def predict(self, X):
        if LANE_D_ARM == "off":
            return super().predict(X)
        if LANE_D_ARM != "output" or self.cf is None or self.cp is None:
            raise RuntimeError("Lane D requires the signed cap250 terminal stages")
        labels = []
        for x in to_windows(X, self.perm):
            p = self.step(x)
            recognised = self.recB if self.routed else not self.active
            raw = self.cp.step(x, p, self.cf.step(x, p), recognised)
            labels.append(int(self.lane_d.step(raw).argmax()))
        return torch.as_tensor(labels, dtype=torch.long)


Ensemble6 = LaneDEnsemble
