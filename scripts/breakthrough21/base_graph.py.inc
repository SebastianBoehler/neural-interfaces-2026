class BaseMemberGraph:
    """Capture frozen and SessionBN forward kernels with fixed CUDA shapes."""

    CAPTURED = (Shallow, FBC, SharedFIRSJit)

    def __init__(self):
        self.graph = self.host = self.static = self.output = None
        self.banks = ()
        self.ids = ()
        self.replays = 0

    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        return clone

    @staticmethod
    def _prefilter(groups, inputs):
        out = []
        for (_, members), tensors in zip(groups, inputs, strict=True):
            shared = next((net for _, net in members if type(net) is SharedFIRSJit), None)
            out.append(None if shared is None else shared.m.spectral_filtering(tensors[shared.dtype]))
        return out

    def _forward(self, groups):
        probs = []
        for (_, members), inputs, fir in zip(groups, self.static, self.banks, strict=True):
            cache = {"sbn_bank": fir} if fir is not None else {}
            for _, net in members:
                if type(net) in self.CAPTURED:
                    probs.append(net.proba(inputs[net.dtype], cache))
        return torch.stack(probs)

    def _capture(self, groups, values, dev):
        with torch.cuda.device(dev):
            self.host = [{dtype: value.pin_memory() for dtype, value in row.items()} for row in values]
            self.static = [{dtype: value.to(dev) for dtype, value in row.items()} for row in self.host]
            self.banks = self._prefilter(groups, self.static)
            current = torch.cuda.current_stream()
            capture = torch.cuda.Stream()
            capture.wait_stream(current)
            try:
                with torch.cuda.stream(capture):
                    for _ in range(3):
                        self._forward(groups)
                    capture.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    graph.capture_begin()
                    try:
                        self.output = self._forward(groups)
                    finally:
                        graph.capture_end()
            finally:
                current.wait_stream(capture)
            for _, members in groups:
                for _, net in members:
                    if type(net) in self.CAPTURED:
                        net.reset_state()
            self.graph = graph
            self.ids = tuple(i for _, members in groups for i, net in members if type(net) in self.CAPTURED)

    def _refresh(self, groups, cpu):
        for values, host, static in zip(cpu, self.host, self.static, strict=True):
            for dtype, source in values.items():
                host[dtype].copy_(source)
                static[dtype].copy_(host[dtype], non_blocking=True)
        for bank, current in zip(self.banks, self._prefilter(groups, self.static), strict=True):
            if bank is not None:
                bank.copy_(current)

    def run(self, bank, x):
        bank.last_inputs = {}
        cpu = []
        for stream, members in bank.groups:
            h = stream(x)
            bank.last_inputs.update({i: h for i, _ in members})
            native = torch.from_numpy(h[None])
            cpu.append({dtype: native.to(dtype=dtype) for dtype in dict.fromkeys(net.dtype for _, net in members)})
        with torch.cuda.device(bank.dev):
            if self.graph is None:
                self._capture(bank.groups, cpu, bank.dev)
            else:
                self._refresh(bank.groups, cpu)
            self.graph.replay()
            self.replays += 1
            cached = dict(zip(self.ids, self.output.unbind(), strict=True))
            ids, probs = [], []
            for (_, members), device in zip(bank.groups, self.static, strict=True):
                for i, net in members:
                    ids.append(i)
                    probs.append(cached[i] if i in cached else net.proba(device[net.dtype], {}))
        return list(zip(ids, torch.stack(probs).cpu().numpy()[:, 0], strict=True))
