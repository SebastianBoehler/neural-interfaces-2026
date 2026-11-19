"""A single full-depth pretrained EEG encoder with a spatial task readout."""
import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, arch, names, positions=None):
        super().__init__()
        self.arch = arch
        common = dict(n_chans=len(names), n_times=800, n_outputs=4, sfreq=200)
        if arch == "reve":
            from braindecode.models import REVE

            self.encoder = REVE(**common)
            self.encoder.final_layer = nn.Identity()
            positions = self.encoder.get_positions(names)
            width = self.encoder.embed_dim
        elif arch == "cbramod":
            from braindecode.models import CBraMod

            self.encoder = CBraMod(**common, return_encoder_output=True)
            width = self.encoder._emb_dim
        elif arch == "manas":
            self.encoder = Manas2()
            width = 512
        else:
            raise ValueError(f"Unknown encoder {arch}")
        self.register_buffer("positions", positions)
        self.token = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU())
        self.spatial = nn.Sequential(
            nn.Flatten(), nn.Linear(len(names) * 64, 160), nn.LayerNorm(160), nn.GELU()
        )
        self.task = nn.Linear(160, 4)

    def features(self, x):
        x = F.interpolate(x, size=800, mode="linear", align_corners=False).clamp(-15, 15)
        if self.arch == "manas":
            tokens = self.encoder(x, self.positions.expand(len(x), -1, -1)).transpose(1, 2)
        elif self.arch == "reve":
            tokens = self.encoder(x, pos=self.positions.unsqueeze(0), return_features=True)["features"]
        else:
            tokens = self.encoder(x * 0.15, return_features=True)["features"]
        tokens = self.token(tokens)
        average = tokens.mean(2)
        progression = tokens[:, :, -2:].mean(2) - tokens[:, :, :2].mean(2)
        return self.spatial(torch.cat((average, progression), -1))
