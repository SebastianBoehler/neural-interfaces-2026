"""Small learned temporal decoder and its training-only masked EEG objective."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from waveform import SENSORS, WIDTH, SessionDecoder

KINDS = ("raw_supervised", "raw_masked")
TOKEN_WIDTH = 48
TOKENS_PER_SPAN = 20


class RawDecoder(SessionDecoder):
    def __init__(self):
        super().__init__()
        self.centered = False
        self.stem = nn.Conv1d(SENSORS, TOKEN_WIDTH, 21, stride=6, padding=10)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(TOKEN_WIDTH, TOKEN_WIDTH, 5, padding=2*dilation,
                                    dilation=dilation, groups=TOKEN_WIDTH, bias=False),
                          nn.Conv1d(TOKEN_WIDTH, TOKEN_WIDTH, 1), nn.GELU())
            for dilation in (1, 3, 9)
        ])
        self.project = nn.Sequential(nn.LayerNorm(TOKEN_WIDTH * 8),
                                     nn.Linear(TOKEN_WIDTH * 8, 64), nn.GELU(),
                                     nn.Linear(64, WIDTH), nn.LayerNorm(WIDTH))
        self.init_head()

    def prepare(self, x):
        return x / 20

    def contextualize(self, tokens):
        for block in self.blocks:
            tokens = tokens + block(tokens)
        return tokens

    def encode(self, eeg):
        tokens = self.contextualize(self.stem(eeg))
        quarters = tokens.reshape(len(eeg), TOKEN_WIDTH, 4, TOKENS_PER_SPAN)
        mean = quarters.mean(dim=-1)
        variation = (quarters - mean.unsqueeze(-1)).square().mean(dim=-1).clamp_min(1e-8).sqrt()
        return self.project(torch.cat((mean, variation), dim=-1).flatten(1))


class MaskedObjective(nn.Module):
    def __init__(self, model, kind, seed, device):
        super().__init__()
        self.update_encoder = kind == "raw_masked"
        self.teacher = deepcopy(model.stem).requires_grad_(False)
        self.head = nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH).to(device)
        self.rng = torch.Generator(device=device).manual_seed(seed + 10000)

    def forward(self, model, eeg):
        starts = torch.randint(0, 61, (len(eeg), 1), generator=self.rng, device=eeg.device)
        selected = starts + torch.arange(TOKENS_PER_SPAN, device=eeg.device)
        samples = torch.arange(480, device=eeg.device)
        mask = (samples >= 6 * starts) & (samples < 6 * (starts + TOKENS_PER_SPAN))
        masked_eeg = eeg.masked_fill(mask[:, None], 0)
        indices = selected[:, None].expand(-1, TOKEN_WIDTH, -1)
        h = model.contextualize(model.stem(masked_eeg)).gather(-1, indices).transpose(1, 2)
        if not self.update_encoder:
            h = h.detach()
        target = self.teacher(eeg).gather(-1, indices).transpose(1, 2)
        return (1 - F.cosine_similarity(self.head(h), target, dim=-1)).mean()

    @torch.no_grad()
    def update(self, model):
        for target, current in zip(self.teacher.parameters(), model.stem.parameters(), strict=True):
            target.lerp_(current, .01)
