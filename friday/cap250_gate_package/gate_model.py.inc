class StockProtectedSolverModel:
    """Attach the measured dual-tail selector only within a predict call."""

    def __init__(self, stock):
        self.stock = stock
        self.selector = None

    def reset_state(self):
        self.stock.reset_state()
        self.selector = None

    def predict(self, x):
        model = self.stock
        if self.selector is None:
            self.selector = LiveSelector(model)
        selected = self.selector
        model.mix6 = selected.mix
        model.routeB.stepx = selected.route
        model.cp.step = selected.terminal
        try:
            return selected.predict(x)
        finally:
            model.mix6 = selected.native_mix
            model.routeB.stepx = selected.native_route
            model.cp.step = selected.native_terminal
