
    def _forward_seeds(self, aligned):
        z = torch.nn.functional.interpolate(aligned, size=800, mode="linear", align_corners=False)
        z = z.clamp(-15, 15) * self.SCALE
        out = []
        for i, (net, head) in enumerate(zip(self.nets, self.heads, strict=True)):
            j = 0 if self.shared else i
            f = net(z[j:j + 1], return_features=True)["features"]
            out.append(head(f.permute(0, 2, 1, 3).flatten(2)).mean(1).softmax(1)[0])
        return torch.stack(out)

    def _posterior(self, x):
        priors = self.priors[:1] if self.shared else self.priors
        aligned = torch.from_numpy(np.stack([self.whiten(x, prior) for prior in priors]))
        if torch.device(self.dev).type == "cuda":
            out = self._forward_graph.run(self._forward_seeds, aligned, self.dev)
        else:
            out = self._forward_seeds(aligned.to(self.dev))
        return out.cpu().numpy().astype(np.float64).mean(axis=0)

    class ForwardGraph:
        """Capture the frozen seed forwards on first use in each session."""

        def __init__(self):
            self.graph = None
            self.failed = False
            self.replays = 0

        def __deepcopy__(self, memo):
            # A graph holds the original model's CUDA addresses.
            clone = type(self)()
            memo[id(self)] = clone
            return clone

        @staticmethod
        def capture(forward, aligned, dev):
            with torch.cuda.device(dev):
                host = torch.empty_like(aligned, pin_memory=True)
                host.copy_(aligned)
                static = host.to(dev)
                current = torch.cuda.current_stream()
                stream = torch.cuda.Stream()
                stream.wait_stream(current)
                try:
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            forward(static)
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        graph.capture_begin()
                        try:
                            output = forward(static)
                        finally:
                            graph.capture_end()
                finally:
                    current.wait_stream(stream)
            return graph, host, static, output

        def run(self, forward, aligned, dev):
            if self.graph is None and not self.failed:
                try:
                    self.graph, self.host, self.static, self.output = self.capture(forward, aligned, dev)
                except RuntimeError as exc:
                    import warnings

                    self.failed = True
                    warnings.warn(f"CBraMod CUDA graph capture failed, using eager: {exc}", RuntimeWarning)
            if self.graph is None:
                return forward(aligned.to(dev))
            self.host.copy_(aligned)
            self.static.copy_(self.host, non_blocking=True)
            self.graph.replay()
            self.replays += 1
            return self.output
