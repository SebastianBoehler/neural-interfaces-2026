
    def _embed_batch(self, x):
        return tuple(net.feats(x) for net in self.nets)

    def _embed(self, x):
        graph = self.feature_graphs.get(len(x))
        if graph is None:
            return self._embed_batch(x)
        return graph.run(self._embed_batch, x, self.device)

    def _embed_prefix(self, xs):
        batches = []
        for i in range(0, len(xs), 64):
            features = self._embed(xs[i:i + 64])
            # A second full chunk reuses the same static graph outputs.
            if 64 in self.feature_graphs and len(xs) - i >= 128:
                features = tuple(f.clone() for f in features)
            batches.append(features)
        return [torch.cat(chunks) for chunks in zip(*batches, strict=True)]
