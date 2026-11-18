"""Maintain the three immutable-kernel histories inside one delivered session."""
from pathlib import Path
import numpy as np
import torch
from fit454 import features
from memory import SpectralMemory


class Actor:
    def __init__(self, seed):
        data = torch.load(Path('/weights')/f'all_{seed}.pt', map_location='cpu', weights_only=True)
        with torch.random.fork_rng(devices=[]):
            self.model = SpectralMemory('kernel')
        self.model.load_state_dict(data['model'], strict=True)
        self.model.cuda().eval()
        self.mean, self.scale, self.centroid, self.donor = [data[k].cuda() for k in ('mean','scale','centroid','donor')]
        assert self.mean.numel() == self.scale.numel() == 246 and self.centroid.numel() == 48
        assert self.donor.ndim == 2 and self.donor.shape[1] == 48 and len(self.donor) > 0
        assert self.scale.min() > 0

    @torch.no_grad()
    def step(self, spectral, states, age):
        z = self.model.encoder(((spectral-self.mean)/self.scale).clamp(-8,8))
        writes = dict(memory=self.model.writes(z),source=self.centroid.reshape(1,48),
                      donor=self.donor[age%len(self.donor)].reshape(1,48))
        logits = {}
        for arm,w in writes.items():
            logits[arm],states[arm] = self.model.sequence(z,w,states.get(arm))
            assert states[arm][1] == age+1
        score = {k:v.detach().cpu().numpy().reshape(3).astype(np.float64) for k,v in logits.items()}
        delta = {k:v-score['source'] for k,v in score.items()}
        assert np.all(delta['source']==0) and all(np.isfinite(v).all() for v in delta.values())
        if age == 0:
            assert all(np.all(v==0) for v in delta.values())
        return {k:.1*np.clip(v-v.mean(),-1,1) for k,v in delta.items()}
