"""Load an authenticated self-contained raw S1C artifact for one session."""
from pathlib import Path

import numpy as np
import torch


class RawS1C:
    def __init__(self, path: str | Path, device):
        self.device = torch.device(device)
        self.module = torch.jit.load(str(path), map_location=self.device).eval()
        self.reset_state()

    @torch.no_grad()
    def reset_state(self):
        if self.module.reset().item() != 0:
            raise ValueError("S1C reset failed")

    @torch.no_grad()
    def predict_proba(self, window):
        value = torch.as_tensor(window, device=self.device)
        if value.shape == (41, 480):
            value = value.unsqueeze(0)
        if value.shape != (1, 41, 480) or value.dtype != torch.float32:
            raise ValueError("raw S1C requires one float32 41-channel, 480-sample window")
        return self.module(value.contiguous()).cpu().numpy()[0]
