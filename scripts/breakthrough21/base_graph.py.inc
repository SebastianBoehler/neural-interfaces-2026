class BaseMemberGraph:
    """Capture each base member forward while preserving its session-local buffer updates."""

    def __init__(self):
        self.graph = self.host = self.static = self.output = None
        self.ids = ()
        self.replays = 0

    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        return clone

    @staticmethod
    def _forward(groups, static):
        probs = []
        for (_, members), inputs in zip(groups, static, strict=True):
            cache = {}
            for _, net in members:
                probs.append(net.proba(inputs[net.dtype], cache))
        return torch.stack(probs)

    def _capture(self, groups, values, dev):
        with torch.cuda.device(dev):
            self.host = [{dtype: value.pin_memory() for dtype, value in row.items()} for row in values]
            self.static = [{dtype: value.to(dev) for dtype, value in row.items()} for row in self.host]
            current = torch.cuda.current_stream()
            capture = torch.cuda.Stream()
            capture.wait_stream(current)
            try:
                with torch.cuda.stream(capture):
                    for _ in range(3):
                        self._forward(groups, self.static)
                    capture.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    graph.capture_begin()
                    try:
                        self.output = self._forward(groups, self.static)
                    finally:
                        graph.capture_end()
            finally:
                current.wait_stream(capture)
            for _, members in groups:
                for _, net in members:
                    net.reset_state()
            self.graph = graph
        self.ids = tuple(i for _, members in groups for i, _ in members)

    def run(self, bank, x):
        bank.last_inputs = {}
        cpu = []
        for stream, members in bank.groups:
            h = stream(x)
            bank.last_inputs.update({i: h for i, _ in members})
            native = torch.from_numpy(h[None])
            cpu.append({dtype: native.to(dtype=dtype) for dtype in dict.fromkeys(net.dtype for _, net in members)})
        if self.graph is None:
            self._capture(bank.groups, cpu, bank.dev)
        with torch.cuda.device(bank.dev):
            for values, host, static in zip(cpu, self.host, self.static, strict=True):
                for dtype, source in values.items():
                    host[dtype].copy_(source)
                    static[dtype].copy_(host[dtype], non_blocking=True)
            self.graph.replay()
            self.replays += 1
        return list(zip(self.ids, self.output.cpu().numpy()[:, 0], strict=True))
