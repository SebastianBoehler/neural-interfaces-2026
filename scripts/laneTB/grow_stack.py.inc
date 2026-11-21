class GrowStack(TeamBlendStack):
    native_weight = 0.0
    terminal_weight = 0.0
    partner = None
    partner_kind = None

    def reset_state(self):
        super().reset_state()
        if self.partner is not None:
            self.partner.reset_state()
        self.grow_calls = 0
        self.grow_move = 0.0

    def _grow_posterior(self, raw):
        self.grow_calls += 1
        return self._grow_prob(self.team_b.posterior(torch.from_numpy(raw)))

    @staticmethod
    def _grow_prob(q):
        if torch.is_tensor(q):
            q = q.detach().cpu().numpy()
        if q.shape != (3,) or not np.isfinite(q).all() or q.min() < 0 or not np.isclose(q.sum(), 1, atol=1e-6):
            raise ValueError("invalid source posterior")
        return q

    def _partner_posterior(self, x):
        label = self.partner.predict(torch.from_numpy(x[None]))
        q = self._grow_prob(self.partner.last)
        if label.shape != (1,) or int(label[0]) != int(q.argmax()):
            raise ValueError("partner prediction differs from captured posterior")
        return q

    def _grow_blend(self, p, q, w):
        result = (1 - w) * p + w * q
        self.grow_move = max(self.grow_move, float(np.max(np.abs(result - p))))
        return result

    def grow_native(self, mix, x):
        if not self.native_weight:
            return mix
        return self._grow_blend(mix, self._grow_posterior(x), self.native_weight)

    def mix6(self, mix, members, p_ft):
        raw = self.hard_window
        if self.partner_kind == "c" and members:
            m = HardWaveEnsemble.mix6(self, mix, members, p_ft)
            q = self._grow_posterior(raw)
            c = self._partner_posterior(raw)
            return self._grow_blend(m, 0.5 * (q + c), self.TEAM_B_MIX)
        if not self.TEAM_B_MIX and self.team_b is not None:
            m = HardWaveEnsemble.mix6(self, mix, members, p_ft)
            if self.native_weight:
                self._grow_prob(m)
            return m
        if self.TEAM_B_MIX and members:
            self.grow_calls += 1
        return super().mix6(mix, members, p_ft)

    def grow_label(self, x, p):
        previous = self.cp.fused
        q = self.cp.step(x, p, self.cf.step(x, p), self.recB if self.routed else not self.active)
        if not self.terminal_weight:
            return int(q.argmax())
        if self.cp.fused != previous:
            s = np.exp(q - q.max())
            q = s / s.sum()
        else:
            q = q / q.sum()
        secondary = self._partner_posterior(x) if self.partner_kind == "a" else self._grow_posterior(x)
        return int(self._grow_blend(self._grow_prob(q), secondary, self.terminal_weight).argmax())
