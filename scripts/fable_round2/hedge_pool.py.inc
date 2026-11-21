    def hedge_pool(self, p_adapted, p_team_b):
        """Causal geometric pool at the existing Team B insertion."""
        w0, w_inf, tau, w_disagree = self.ROUND2_HEDGE
        if (p_adapted < 0).any() or not np.isfinite(p_adapted).all():
            raise ValueError("invalid adapted posterior for Team B hedge")
        self.hedge_seen += 1
        self.hedge_agreed += int(p_adapted.argmax() == p_team_b.argmax())
        agreement = self.hedge_agreed / self.hedge_seen
        w = min(1.0, max(0.0, w_inf + (w0 - w_inf) * np.exp(-self.hedge_seen / tau)
                         + w_disagree * (1.0 - agreement)))
        return softmax((1 - w) * log_floor(p_adapted) + w * log_floor(p_team_b))
