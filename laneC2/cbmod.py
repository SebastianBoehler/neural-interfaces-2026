"""Lane C2: one CBraMod encoder with the Lane C member interface and montage-specific flattened heads.

The encoder is shared across corpora; each distinct montage gets its own task head and motor-subtype head on the flattened
channel-flattened tokens of each 200-sample patch (the §280 head form, weights shared over patches so that 2- and 4-patch
corpora fit one head). Two pooled heads (mean over tokens, then a coordinate-aware
per-token head) stayed at the constant posterior (C2 diagnostics 64853666, 64855209, 64856291), so they are retired.
"""
import zlib

import torch
from torch import nn
from torch.nn import functional as F

from model import checkpoint_tensors  # the venv has no safetensors package
from oracle import require, sha

BLOB = "a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3"
EMB = 200
SCALE = 0.15  # experiments/deep.py PRETRAINED["cbramod"]: unit-RMS aligned input -> CBraMod's 100 uV units
FLAT = {"task": 4, "motor": 5}


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        from braindecode.models import CBraMod

        self.net = CBraMod(n_outputs=4, n_chans=41, n_times=800, sfreq=200, return_encoder_output=True)


class Member(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        require(sha(checkpoint) == BLOB, "CBraMod checkpoint differs from the pinned §280 blob")
        self.body = Body()
        load = self.body.net.load_state_dict(checkpoint_tensors(checkpoint), strict=False)
        require(not load.unexpected_keys and not load.missing_keys, f"incomplete CBraMod load: {load}")
        self.heads = nn.ModuleDict()

    def _head(self, key, width):
        if key not in self.heads:
            self.heads[key] = nn.ModuleDict({k: nn.Linear(width, n) for k, n in FLAT.items()})
        return self.heads[key]

    def load_state_dict(self, state, strict=True, **kw):
        for name, tensor in state.items():  # heads exist per montage; create them before loading
            if name.startswith("heads.") and name.endswith(".weight"):
                self._head(name.split(".")[1], tensor.shape[1])
        return super().load_state_dict(state, strict=strict, **kw)

    def positions(self, names, device):
        """A montage tag, not coordinates (CBraMod has none). It selects this montage's heads, created here so that
        the optimizer sees them. Heads act per 200-sample patch on the channel-flattened tokens and average over patches."""
        tag = zlib.crc32(",".join(names).encode()) % (1 << 20)
        self._head(f"h{tag}", len(names) * EMB).to(device)
        return torch.full((1, len(names), 3), float(tag), device=device)

    def forward(self, x, pos):
        x = F.interpolate(x, size=round(x.shape[-1] * 200 / 120), mode="linear", align_corners=False)
        feats = self.body.net(x.clamp(-15, 15) * SCALE, return_features=True)["features"]  # B, C, P, 200
        feats = feats.permute(0, 2, 1, 3).flatten(2)
        head = self.heads[f"h{int(pos[0, 0, 0].item())}"]
        return head["task"](feats).mean(1), head["motor"](feats).mean(1)


def augment(x, pos, rng):
    """Lane C's source-only augmentation with sensor dropout as zeroing (flattened heads need a fixed montage)."""
    keep = torch.as_tensor(rng.random(x.shape[1]) >= 0.12, device=x.device)
    if not bool(keep.any()):
        raise RuntimeError("augmentation removed every electrode")
    x = x * keep[None, :, None]
    shift = int(rng.integers(-18, 19))
    if shift:
        x = F.pad(x, (max(shift, 0), max(-shift, 0)), mode="reflect")
        x = x[..., :x.shape[-1]-shift] if shift > 0 else x[..., -shift:]
    scale = torch.empty((len(x), 1, 1), device=x.device).uniform_(0.75, 1.25)
    return x * scale + 0.05 * torch.randn_like(x), pos


def split_optimizer(model):
    head = list(model.heads.parameters())
    head_set = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_set]
    require(head and body and len(body) + len(head) == len(list(model.parameters())), "optimizer groups do not partition")
    return torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
