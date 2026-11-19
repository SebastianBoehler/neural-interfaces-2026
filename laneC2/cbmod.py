"""Lane C2: one CBraMod encoder with the Lane C member interface (masked task head, motor-subtype head)."""
import torch
from torch import nn
from torch.nn import functional as F

from model import checkpoint_tensors  # the venv has no safetensors package
from oracle import require, sha

BLOB = "a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3"
EMB = 200
SCALE = 0.15  # experiments/deep.py PRETRAINED["cbramod"]: unit-RMS aligned input -> CBraMod's 100 uV units


class Body(nn.Module):
    """Holds the encoder and heads under the names Lane C's optimizer groups expect."""

    def __init__(self):
        super().__init__()
        from braindecode.models import CBraMod

        self.net = CBraMod(n_outputs=4, n_chans=41, n_times=800, sfreq=200, return_encoder_output=True)
        self.final_layer = nn.Sequential(nn.LayerNorm(EMB), nn.Linear(EMB, 4))


class Member(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        require(sha(checkpoint) == BLOB, "CBraMod checkpoint differs from the pinned §280 blob")
        self.body = Body()
        load = self.body.net.load_state_dict(checkpoint_tensors(checkpoint), strict=False)
        require(not load.unexpected_keys and not load.missing_keys, f"incomplete CBraMod load: {load}")
        self.motor = nn.Linear(EMB, 5)

    def positions(self, names, device):
        return torch.zeros(1, len(names), 3, device=device)  # CBraMod has no electrode coordinates

    def forward(self, x, pos):
        x = F.interpolate(x, size=round(x.shape[-1] * 200 / 120), mode="linear", align_corners=False)
        feats = self.body.net(x.clamp(-15, 15) * SCALE, return_features=True)["features"]
        pooled = feats.mean((1, 2))
        return self.body.final_layer(pooled), self.motor(F.layer_norm(pooled, (EMB,)))


def split_optimizer(model):
    head = list(model.body.final_layer.parameters()) + list(model.motor.parameters())
    head_set = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_set]
    require(body and all(id(p) not in head_set for p in body) and len(body) + len(head) == len(list(model.parameters())),
            "optimizer groups do not partition the parameters")
    return torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
