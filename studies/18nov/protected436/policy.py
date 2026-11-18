class ProtectedEnsemble(ReuseEnsemble):
    """Keep a baseline outer history beside JS. The unchanged inner gates select the output."""

    def __init__(self, z, dev, root):
        import copy
        super().__init__(z, dev, root)
        if not self.js_on or self.cp is None or self.cf is None:
            raise ValueError("protected policy requires the pinned JS, count and personalization stages")
        self.base_route = copy.deepcopy(self.routeB)
        self.base_route.a = self.w[0] * (1 - self.w_rest)
        self.base_filter = copy.deepcopy(self.cf)
        self.base_personal = copy.deepcopy(self.cp)
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        if hasattr(self, "base_route"):
            for stage in (self.base_route, self.base_filter, self.base_personal):
                stage.reset_state()
        self.output436 = None

    def predict(self, X):
        labels = []
        for x in to_windows(X, self.perm):
            js_raw = self.step(x)
            if self.dead or self.ft.over:
                raise RuntimeError("protected policy lost its common inner path")
            trace = self.trace
            outer = trace["v6"]
            mix = Ensemble6.mix6(self, trace["mix"], outer["members"], outer["p_ft"])
            raw = self.base_route.stepx(x, trace["u"], mix, trace["core1"], outer["extra"],
                                        trace["contrast"], trace["identity"])
            baseline = self.base_personal.step(x, raw, self.base_filter.step(x, raw), self.recB)
            js = self.cp.step(x, js_raw, self.cf.step(x, js_raw), self.recB)
            if any(stage.dead for stage in (self.cp, self.cf, self.base_filter, self.base_personal)):
                raise RuntimeError("protected outer history failed")
            self.output436 = baseline if self.recB else js
            labels.append(int(self.output436.argmax()))
        return torch.as_tensor(labels, dtype=torch.long)
