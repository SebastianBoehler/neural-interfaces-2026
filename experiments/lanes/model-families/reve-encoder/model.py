"""One REVE-Base encoder with a masked task head and motor-subtype supervision."""
import json
import struct

import torch
from torch import nn
from torch.nn import functional as F

from oracle import require, sha

BLOB = "8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe"


def checkpoint_tensors(path):
    # The pinned §280 REVE export contains float32 tensors exclusively.
    with path.open("rb") as fd:
        header_size, = struct.unpack("<Q", fd.read(8))
        header = json.loads(fd.read(header_size))
        data = memoryview(fd.read())
    tensors = {}
    for name, descriptor in header.items():
        if name == "__metadata__" or name.startswith("final_layer."):
            continue
        require(descriptor["dtype"] == "F32", f"unsupported checkpoint tensor type: {name}")
        start, stop = descriptor["data_offsets"]
        tensors[name] = torch.frombuffer(bytearray(data[start:stop]), dtype=torch.float32).reshape(descriptor["shape"])
    return tensors


class Member(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from braindecode.models import REVE

        require(sha(checkpoint) == BLOB, "REVE base checkpoint differs from §280")
        self.body = REVE(n_outputs=4, n_chans=41, n_times=800, sfreq=200, attention_pooling=True)
        load = self.body.load_state_dict(checkpoint_tensors(checkpoint), strict=False)
        require(not load.unexpected_keys and all(k.startswith("final_layer.") for k in load.missing_keys),
                f"incomplete REVE load: {load}")
        self.motor = nn.Linear(self.body.embed_dim, 5)

    def positions(self, names, device):
        return self.body.get_positions(names).reshape(1, len(names), 3).to(device)

    def forward(self, x, pos):
        x = F.interpolate(x, size=round(x.shape[-1] * 200 / 120), mode="linear", align_corners=False)
        features = self.body(x.clamp(-15, 15), pos=pos.expand(len(x), -1, -1), return_features=True)["features"]
        pooled = self.body._attention_pooling(features)
        return self.body.final_layer(pooled), self.motor(pooled)


def augment(x, pos, rng):
    # One montage for the batch, independent amplitudes/noise per window.
    keep = torch.as_tensor(rng.random(x.shape[1]) >= 0.12, device=x.device)
    if not bool(keep.any()):
        raise RuntimeError("augmentation removed every electrode")
    x = x[:, keep]
    shift = int(rng.integers(-18, 19))
    if shift:
        x = F.pad(x, (max(shift, 0), max(-shift, 0)), mode="reflect")
        x = x[..., :x.shape[-1]-shift] if shift > 0 else x[..., -shift:]
    scale = torch.empty((len(x), 1, 1), device=x.device).uniform_(0.75, 1.25)
    return x * scale + 0.05 * torch.randn_like(x), pos[:, keep]


def masked_ce(logits, y, valid):
    labels = torch.as_tensor(valid, device=y.device)
    target = (y[:, None] == labels).long().argmax(1)
    return F.cross_entropy(logits[:, labels], target)
