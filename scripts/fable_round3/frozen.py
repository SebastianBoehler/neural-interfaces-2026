"""Analysis-only read queries against one complete session checkpoint."""
import numpy as np
import torch

from variants import predict


def containers(value):
    if isinstance(value, list):
        return [containers(v) for v in value]
    if isinstance(value, dict):
        return {k: containers(v) for k, v in value.items()}
    return value


def parts(model):
    yield from (model, model.core1, model.core2, model.nets, model.eye, model.rest, model.rest.stream,
                model.ft, model.cf, model.cp, model.team_b)
    for stream, members in model.nets.groups:
        yield stream
        yield from (net for _, net in members)
    for route in (model.route, model.routeB):
        yield from (route, route.contrast, route.identity, route.own)
    for tail in (model.shadow, model.route.tail, model.routeB.tail, model.tail6, model.tail6r):
        if tail is not None:
            yield from (tail, tail.rb, tail.em, tail.tsem)


def script_buffers(model):
    nets = [net.m for _, members in model.nets.groups for _, net in members if hasattr(net, "m")]
    nets += list(model.rest.nets) + list(model.hard_actors)
    if model.ft.nets is not None:
        nets.extend(model.ft.nets.values())
    for net in nets:
        for name, buffer in net.named_buffers():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("s1", "s2", "n") or leaf.startswith("hist_"):
                yield buffer


class Checkpoint:
    def __init__(self, model):
        if (model.hard_actors is None or model.team_b.cov is None or
                len(model.routeB.tail.em.A) != model.log6["v6"]):
            raise ValueError("incomplete end-of-stream state")
        self.model = model
        contrast, identity = (model.route.last[k] for k in ("contrast", "identity"))
        model.route.contrast.step = lambda *_: contrast
        model.route.identity.step = lambda *_: identity
        model.ft.refit = refuse_refit
        model.cp.refit = lambda *_: model.cp.model
        self.state = tuple((part, containers(vars(part))) for part in parts(model))
        self.mutable_arrays = [(part, key, getattr(part, key).copy())
                               for part, key in ((model.core1, "log_sum"), (model.core2, "log_sum"),
                                                 (model.cf, "al"))]
        self.buffers = [(buf, buf.detach().clone()) for buf in script_buffers(model)]
        if not self.buffers:
            raise ValueError("missing scripted session statistics")

    @torch.no_grad()
    def restore(self):
        for part, state in self.state:
            vars(part).clear()
            vars(part).update(containers(state))
        for part, key, value in self.mutable_arrays:
            setattr(part, key, value.copy())
        for buffer, value in self.buffers:
            buffer.copy_(value)

    @torch.no_grad()
    def label(self, window):
        try:
            return predict(self.model, window[None])
        finally:
            self.restore()

    def verify(self):
        for part, key, value in self.mutable_arrays:
            np.testing.assert_array_equal(getattr(part, key), value)
        for live, saved in self.buffers:
            assert torch.equal(live, saved)
        assert len(self.model.team_b.windows) == self.model.log6["v6"]
        assert len(self.model.routeB.tail.em.A) == self.model.log6["v6"]

    def close(self):
        del self.model.route.contrast.step, self.model.route.identity.step
        del self.model.ft.refit, self.model.cp.refit


def refuse_refit(*_):
    # The endpoint fine-tuned parameters are already loaded for an eligible stream.
    return None


def frozen_labels(model, windows, smoke=False):
    checkpoint = Checkpoint(model)
    try:
        result = np.fromiter((checkpoint.label(x) for x in windows), dtype=np.int8, count=len(windows))
        if smoke:
            replay = np.fromiter((checkpoint.label(x) for x in reversed(windows)), dtype=np.int8, count=len(windows))
            np.testing.assert_array_equal(result, replay[::-1])
            checkpoint.verify()
        return result
    finally:
        checkpoint.close()
