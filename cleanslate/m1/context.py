"""Coordinate-invariant set attention over a session's delivered EEG embeddings."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

MAX_PREFIX = 190


def capture(nets, forward):
    """Read each member's penultimate vector and output from one ordinary forward pass."""
    records = [None] * len(nets)

    def hook(index):
        def collect(_module, args, logits):
            z = args[0].float()
            z = F.normalize(z - z.mean(-1, keepdim=True), dim=-1)
            records[index] = (z, logits.float())
        return collect

    handles = [net.fc.register_forward_hook(hook(i)) for i, net in enumerate(nets)]
    try:
        result = forward()
    finally:
        for handle in handles:
            handle.remove()
    if any(record is None for record in records):
        raise RuntimeError("a C1 member skipped the embedding head")
    vectors, logits = zip(*records)
    return result, vectors, torch.stack(logits, dim=1)


class Context(nn.Module):
    def __init__(self, members, heads=4, width=64):
        super().__init__()
        self.heads = heads
        self.scale = math.sqrt(width // heads)
        self.tokens = nn.Sequential(nn.Linear(3 * members, width), nn.LayerNorm(width), nn.GELU())
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.pair_bias = nn.Linear(members, heads, bias=False)
        self.mix = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.target = nn.Linear(width, 3)
        nn.init.zeros_(self.target.weight)
        nn.init.zeros_(self.target.bias)

    def forward(self, current, past, similarity, valid):
        """Return the current token readout after query-only attention over valid earlier tokens."""
        b, t = past.shape[:2]
        q = self.tokens(F.log_softmax(current, dim=-1).flatten(1))
        p = self.tokens(F.log_softmax(past, dim=-1).flatten(2))
        query = self.query(q).reshape(b, self.heads, -1)
        key = self.key(p).reshape(b, t, self.heads, -1)
        value = self.value(p).reshape(b, t, self.heads, -1)
        score = torch.einsum("bhd,bthd->bth", query, key) / self.scale + self.pair_bias(similarity)
        attention = score.masked_fill(~valid[:, :, None], -1e4).softmax(dim=1)
        pooled = torch.einsum("bth,bthd->bhd", attention * valid[:, :, None], value).flatten(1)
        return self.mix(torch.cat((q, pooled), dim=-1))

    def correction(self, current, past, similarity, valid):
        return self.target(self(current, past, similarity, valid)) * valid.any(dim=1)[:, None]


class ContextStream:
    def __init__(self, base, context, use_context):
        self.base, self.context, self.use_context = base, context, use_context
        self.reset_state()

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.base, memo), self.context, self.use_context)

    def reset_state(self):
        self.base.reset_state()
        self.memory = None
        self.previous = None
        self.count = 0
        self.last = None

    @torch.no_grad()
    def predict(self, x):
        label, vectors, logits = capture(self.base.nets, lambda: self.base.predict(x))
        self.last = self.base.last.log()
        if not self.use_context:
            return label
        if self.memory is None:
            self.memory = [z.new_empty((MAX_PREFIX, z.shape[-1])) for z in vectors]
            self.previous = logits.new_empty((MAX_PREFIX, *logits.shape[1:]))
        size = min(self.count, MAX_PREFIX)
        if size:
            similarities = torch.stack([z @ old[:size].T for z, old in zip(vectors, self.memory)], dim=-1)
            valid = torch.ones((1, size), dtype=torch.bool, device=logits.device)
            self.last = self.last + self.context.correction(
                logits, self.previous[None, :size], similarities, valid,
            )[0]
            label = self.last.argmax().reshape(1).cpu()
        slot = self.count % MAX_PREFIX
        for z, old in zip(vectors, self.memory):
            old[slot] = z[0]
        self.previous[slot] = logits[0]
        self.count += 1
        return label
