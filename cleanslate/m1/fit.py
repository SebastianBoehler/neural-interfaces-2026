"""Fit the shared set-attention readout and use one auxiliary head per external corpus."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from context import Context
from episodes import DEVICE

SEED = 1675


def sample(banks, rng):
    bank = banks[int(rng.integers(len(banks)))]
    episode = bank[int(rng.integers(len(bank)))]
    return episode, int(rng.integers(len(episode.labels)))


def batch(examples, output):
    queries, past, sim, valid, labels = output
    for i, (episode, age) in enumerate(examples):
        queries[i] = episode.logits[age]
        past[i, :age] = episode.logits[:age]
        sim[i, :age] = episode.similarity[age, :age]
        valid[i, :age] = True
        labels[i] = int(episode.labels[age])


def train(target, external, classes, members, steps):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Context(members).to(DEVICE)
    heads = nn.ModuleDict({name: nn.Linear(model.target.in_features, k) for name, k in classes.items()}).to(DEVICE)
    opt = torch.optim.AdamW([*model.parameters(), *heads.parameters()], lr=1e-3, weight_decay=0.01)
    names = list(heads)
    pools = list(target.values())
    visited = {k: 0 for k in names}
    total_loss = torch.zeros((), device=DEVICE)
    for step in range(steps):
        targets = [sample(pools, rng) for _ in range(8)]
        sources = [names[i] for i in rng.integers(len(names), size=8)]
        extras = [sample([external[name]], rng) for name in sources]
        examples = targets + extras
        length = max(1, max(age for _, age in examples))
        shape = (16, length)
        output = (
            torch.empty(16, members, 3), torch.zeros(*shape, members, 3),
            torch.zeros(*shape, members), torch.zeros(shape, dtype=torch.bool),
            torch.empty(16, dtype=torch.long),
        )
        batch(examples, output)
        q, p, sim, valid, y = (tensor.to(DEVICE) for tensor in output)
        features = model(q, p, sim, valid)
        anchor = q[:8].softmax(-1).mean(1).log()
        delta = model.target(features[:8]) * valid[:8].any(dim=1)[:, None]
        target_loss = F.cross_entropy(anchor + delta, y[:8])
        other_loss = []
        for i, name in enumerate(sources, 8):
            other_loss.append(F.cross_entropy(heads[name](features[i:i + 1]), y[i:i + 1]))
            visited[name] += 1
        loss = 0.5 * (target_loss + torch.stack(other_loss).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_loss += loss.detach()
        if (step + 1) % 300 == 0 or step + 1 == steps:
            print(f"context step {step + 1}/{steps} mean loss {float(total_loss) / (step + 1):.4f}", flush=True)
    if not torch.isfinite(total_loss) or not model.target.weight.detach().abs().max() > 0:
        raise RuntimeError("context optimization failed")
    if not all(visited.values()):
        raise RuntimeError(f"missing external training heads: {visited}")
    return model.eval(), visited
