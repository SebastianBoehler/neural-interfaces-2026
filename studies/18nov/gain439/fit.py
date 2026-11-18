"""Apply source-derived sensor gains while fitting the pinned whole waveform decoder."""
import os
from pathlib import Path
import sys

import numpy as np
import torch

import train
from model import Decoder
from source_labels import read_training_labels


def source_gains(person, steps):
    with np.load(train.DATA) as data:
        people = np.array([s.split('/')[-1] for s in data['subject']])
        rows = np.flatnonzero(people != person)
        session = np.char.add(np.char.add(people[rows], '|'), data['session'][rows])
        labels = read_training_labels(train.DATA, rows, len(people))
        assert len(set(people[rows])) == 9 and person not in people[rows]
        eeg = data['X'][rows]
    power = np.einsum('nct,nct->nc', eeg, eeg) / eeg.shape[-1]
    log_rms = .5 * np.log(power.clip(1e-16))
    templates = np.stack([np.mean([log_rms[(session == s) & (labels == k)].mean(0)
                                   for k in range(3)], axis=0) for s in np.unique(session)])
    rng = np.random.default_rng(439)
    patterns = np.zeros((steps, 41), dtype=np.float32)
    for episode in range(1, steps, 2):
        a, b = rng.choice(len(templates), 2, replace=False)
        diff = .5 * (templates[a] - templates[b])
        patterns[episode] = np.clip(diff - diff.mean(), -np.log(2), np.log(2))
    coherent = np.repeat(np.arange(steps)[:, None], 96, axis=1)
    independent = coherent.copy()
    for start in range(0, steps, 12):
        block = independent[start:start + 12]
        block[:] = rng.permutation(block.flatten()).reshape(block.shape)
    assert np.array_equal(np.bincount(independent.ravel()), np.bincount(coherent.ravel()))
    gains = np.exp(patterns)
    assert np.isfinite(gains).all() and np.max(abs(gains-1)) > .03
    assert not np.array_equal(gains[independent], gains[coherent])
    return torch.from_numpy(gains).cuda(), (coherent, independent)


def run(person, arm, smoke, init_seed=379, decoder_cls=Decoder):
    prior = Path(os.environ['WAVE379'])
    steps = 2 if smoke else 360
    gains, schedules = source_gains(person, steps)
    assert arm in ('clean', 'coherent', 'independent')
    schedule = torch.from_numpy(schedules[int(arm == 'independent')]).cuda()
    class GainDecoder(decoder_cls):
        def __init__(self):
            torch.manual_seed(init_seed)
            super().__init__()
            self.episode = 0

        def step(self, x, kind, state=None):
            if self.training:
                assert len(x) == 96 and state is None
                if arm != 'clean':
                    x = x * gains[schedule[self.episode], :, None]
                self.episode += 1
            else:
                assert self.episode == steps
            return super().step(x, kind, state)

    train.Decoder = GainDecoder
    train.main(person, smoke)
    if not smoke:
        import json
        proof = json.loads(Path(person+'.json').read_text())
        original = json.loads((prior/(person+'.json')).read_text())
        assert proof['source_episodes'] == original['source_episodes']
        assert proof['train_rows_sha256'] == original['train_rows_sha256']


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3:] == ['smoke'])
