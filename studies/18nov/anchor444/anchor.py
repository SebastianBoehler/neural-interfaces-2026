"""Require a population-typical prefix to map to the same readout as blank history."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from model import Decoder
import fit
import train


class AnchoredRead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.register_buffer('reference', torch.zeros(48))
        self.active = False

    def forward(self, context):
        own = self.original(context)
        if not self.active:
            return own
        age = context[:, -1:]
        typical = torch.cat([self.reference.detach().clone().expand(len(age), -1) * (age > 0), age], 1)
        blank = self.original(torch.zeros_like(context))
        return (own - self.original(typical)) + blank


class AnchoredDecoder(Decoder):
    def __init__(self):
        super().__init__()
        self.read = AnchoredRead(self.read)

    def sequence(self, features, arm, initial=None, donor=None):
        self.read.active = arm == 'memory'
        logits, state = super().sequence(features, arm, initial, donor)
        if self.training and self.read.active:
            with torch.no_grad():
                writes = state[0][:, -48:] / state[1]
                self.read.reference.lerp_(writes[0], .04)
        return logits, state


def main(p, seed, smoke):
    fit.run(p, 'coherent', smoke, seed, AnchoredDecoder)
    if smoke:
        return
    matched = Path('../..') / ('ni26_gain439_20261118' if seed==379 else 'ni26_gain441_20261118') / 'coherent'
    proof = json.loads((matched/(p+'.json')).read_text())
    assert train.sha(matched/(p+'.npz')) == proof['outputs_sha256']
    with np.load(p+'.npz') as a, np.load(matched/(p+'.npz')) as b:
        for arm in ('static','norm'):
            for order in train.ORDERS:
                assert np.array_equal(a[arm+'|'+order], b[arm+'|'+order])
    net = AnchoredDecoder().cuda().eval()
    net.load_state_dict(torch.load(p+'_memory.pt',weights_only=True))
    with np.load(train.DATA) as z:
        people = np.array([s.split('/')[-1] for s in z['subject']])
        samples = torch.from_numpy(z['X']).cuda()
        x = samples[people==p][:24] / samples[people!=p].square().mean().sqrt()
    with torch.no_grad():
        features = net.encode(x)
        typical = torch.zeros(len(x),144,device=x.device)
        typical[:,-48:] = net.read.reference
        assert net.read.reference.abs().max() > 1e-4
        baseline = net.sequence(features,'static')[0]
        anchored = net.sequence(features,'memory',donor=typical)[0]
        err = float((baseline-anchored).abs().max())
        assert err < 2e-5 and torch.equal(baseline.argmax(1),anchored.argmax(1))
    Path(p+'.parity.json').write_text(json.dumps(dict(held=p,seed=seed,static_norm=True,
        max_anchor_error=err,matched=proof['outputs_sha256'])))


if __name__ == '__main__':
    main(sys.argv[1],int(sys.argv[2]),sys.argv[3:] == ['smoke'])
