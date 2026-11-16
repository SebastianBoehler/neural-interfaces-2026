import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factor import FactorNet, contrastive, person_batches


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd = {}
        self.net = nn.Module()
        self.net.final_layer = nn.Sequential(nn.Flatten(), nn.Linear(24, 3))

    def forward(self, x):
        if self.fwd.get("return_features"):
            return {"features": x}
        return self.net.final_layer(x)


def test_off_preserves_backbone_and_split_has_matched_capacity():
    torch.manual_seed(1)
    backbone = ToyBackbone()
    x = torch.randn(4, 2, 3, 4)
    expected = backbone(x)
    model = FactorNet(backbone, "off", dim=4)
    assert torch.equal(model(x)[0], expected)
    torch.manual_seed(2)
    matched = FactorNet(ToyBackbone(), "matched", dim=4, width=2)
    torch.manual_seed(2)
    split = FactorNet(ToyBackbone(), "split", dim=4, width=2)
    assert sum(p.numel() for p in matched.parameters()) == sum(p.numel() for p in split.parameters())
    assert all(torch.equal(a, b) for a, b in zip(matched.parameters(), split.parameters()))


def test_split_loss_updates_both_branches_and_rejects_missing_pairs():
    torch.manual_seed(4)
    model = FactorNet(ToyBackbone(), "split", dim=4, width=2)
    logits, auxiliary = model(torch.randn(6, 2, 3, 4), torch.tensor([0, 0, 0, 1, 1, 1]))
    (logits.square().mean() + auxiliary).backward()
    for branch in (model.task, model.style):
        assert sum(float(p.grad.abs().sum()) for p in branch.parameters()) > 0
    with pytest.raises(ValueError):
        contrastive(torch.randn(3, 4), torch.arange(3))
    with pytest.raises(ValueError):
        contrastive(torch.randn(3, 4), torch.zeros(3))


def test_sampler_has_positive_and_negative_pairs_and_covers_rows():
    people = np.repeat(np.arange(3), [20, 17, 2])
    streams = np.array([f"{p}-{i // 9}" for i, p in enumerate(people)])
    batches = person_batches(np.arange(len(people)), people, streams, np.random.default_rng(7), chunk=4)
    assert set(np.concatenate(batches)) == set(range(len(people)))
    for batch in batches:
        assert len(batch) == 8
        assert len(set(people[batch])) == 2
        assert len(set(streams[batch[:4]])) == len(set(streams[batch[4:]])) == 1


def test_sampler_rejects_single_person():
    with pytest.raises(ValueError):
        person_batches(np.arange(3), np.zeros(3), np.zeros(3), np.random.default_rng(1), chunk=2)
