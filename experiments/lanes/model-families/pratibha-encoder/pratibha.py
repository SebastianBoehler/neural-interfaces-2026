"""Lane C3: Pratibha-2 / MANAS-2 encoder (MannasAI/pratibha-2) with Lane C's member interface.

The encoder classes are a torch-only copy of the audited `model.py` at the pinned revision (the Leonardo venv has no
`transformers`, compute nodes have no network, and `trust_remote_code` is avoided altogether); weights and the electrode
table are the pinned release files and are hash-checked. The released `model.py` imports only json, math, pathlib, torch,
transformers and (lazily) safetensors. Head: learned-query attention pooling over all (patch, channel) tokens, then task
and motor-subtype linear layers (the REVE form; variable montages and 2- or 4-patch crops forbid a flattened head).
"""
import json
import math
import struct
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from oracle import require, sha

REVISION = "863edc940d82a02744776a7820641bf1bc32310b"
HASHES = {"model.safetensors": "5c7fb0a1de75bf01c179fdc416a4e4af3ad4ead7895ac03b7e357adb587cfa64",
          "positions.json": "e7fb3cbd81a7f20b8e2714a4c01681d250fb17b8984e1443735ae8e3ced062d5",
          "model.py": "a637e4143fa271c64dc45c29c8b7eb8fa87b207863e92e0db237d130aa4ffae6"}
BODY_PREFIXES = ("patch_embed.", "pos_enc.", "encoder.")
TRAINING_ONLY = ("decoder.", "band_decoder.", "band_target_module.", "band_pooler.", "aux_linear.", "aux_predict.")


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * self.weight


class GEGLU(nn.Module):
    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        gelu = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (gate + 0.044715 * gate.pow(3))))
        return value * (gate * torch.sigmoid(gate) + gate * gelu)  # the checkpoint's own gate, both terms


class Block(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.heads = heads
        self.pre_attn_norm, self.pre_ffn_norm = RMSNorm(dim), RMSNorm(dim)
        self.attn = nn.ModuleDict(dict(qkv_proj=nn.Linear(dim, 3 * dim), out_proj=nn.Linear(dim, dim)))
        self.ffn = nn.ModuleDict(dict(in_proj=nn.Linear(dim, 8 * dim), gate=GEGLU(), out_proj=nn.Linear(4 * dim, dim)))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [u.view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for u in self.attn["qkv_proj"](self.pre_attn_norm(x)).chunk(3, dim=-1)]
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.attn["out_proj"](a)
        return x + self.ffn["out_proj"](self.ffn["gate"](self.ffn["in_proj"](self.pre_ffn_norm(x))))


class Encoder(nn.Module):
    def __init__(self, dim=512, depth=22, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.final_norm = RMSNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(200, 512, bias=False)

    def forward(self, eeg):
        return self.linear(eeg.unfold(-1, 200, 180))


class PosEnc(nn.Module):
    def __init__(self):
        super().__init__()
        freqs = torch.linspace(1.0, 10.0, 4)
        self.register_buffer("freq_matrix", torch.cartesian_prod(*([freqs] * 4)).T)
        self.fourier_linear = nn.Linear(512, 512, bias=False)
        self.learned_linear = nn.Sequential(nn.Linear(4, 1024, bias=False), GEGLU(), RMSNorm(512))
        self.final_norm = RMSNorm(512)

    def forward(self, coords):
        with torch.autocast(device_type=coords.device.type, enabled=False):  # phases reach ~1e2: never bf16
            coords = coords.float()
            phases = coords @ self.freq_matrix.T.contiguous().T
            fourier = self.fourier_linear(torch.cat([phases.sin(), phases.cos()], dim=-1))
            return self.final_norm(fourier + self.learned_linear(coords))


class Manas2(nn.Module):
    """[B, C, T] and [B, C, 3] positions in cm -> [B, patches, C, 512] (200-sample patches, stride 180)."""

    def __init__(self):
        super().__init__()
        self.patch_embed, self.pos_enc, self.encoder = PatchEmbed(), PosEnc(), Encoder()

    def forward(self, eeg, positions):
        require(eeg.ndim == 3 and eeg.shape[-1] >= 200 and tuple(positions.shape) == (*eeg.shape[:2], 3),
                "bad Pratibha input shape")
        batch, channels, _ = eeg.shape
        tokens = self.patch_embed(eeg)
        patches = tokens.shape[2]
        spatial = positions.unsqueeze(2).expand(-1, -1, patches, -1)
        time = torch.arange(patches, device=eeg.device, dtype=torch.float32).view(1, 1, patches, 1)
        coords = torch.cat([spatial, time.expand(batch, channels, -1, -1)], dim=-1).reshape(batch, channels * patches, 4)
        encoded = self.encoder(tokens.flatten(1, 2) + self.pos_enc(coords))
        return encoded.reshape(batch, channels, patches, 512).permute(0, 2, 1, 3).contiguous()


def release_tensors(path):
    """Body tensors of the pinned release; any other key than a known training-only one is a failure."""
    with path.open("rb") as fd:
        size, = struct.unpack("<Q", fd.read(8))
        header = json.loads(fd.read(size))
        data = memoryview(fd.read())
    out = {}
    for name, d in header.items():
        if name == "__metadata__":
            continue
        if name.startswith(BODY_PREFIXES):
            require(d["dtype"] == "F32", f"unsupported tensor type: {name}")
            a, b = d["data_offsets"]
            out[name] = torch.frombuffer(bytearray(data[a:b]), dtype=torch.float32).reshape(d["shape"])
        else:
            require(name == "aux_query" or name.startswith(TRAINING_ONLY), f"unexpected checkpoint tensor: {name}")
    return out


class Member(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        checkpoint = Path(checkpoint)
        for name, digest in HASHES.items():
            require(sha(checkpoint.parent / name) == digest, f"{name} differs from pinned revision {REVISION}")
        require(checkpoint.name == "model.safetensors", "checkpoint must be the pinned model.safetensors")
        self.table = json.loads((checkpoint.parent / "positions.json").read_text())
        self.body = Manas2()
        state = release_tensors(checkpoint)
        load = self.body.load_state_dict(state, strict=False)  # reported, then required empty
        require(not load.missing_keys and not load.unexpected_keys, f"incomplete Pratibha load: {load}")
        self.score = nn.Linear(512, 1)
        self.task = nn.Linear(512, 4)
        self.motor = nn.Linear(512, 5)

    def positions(self, names, device):
        missing = [n for n in names if n not in self.table]
        require(not missing, f"electrodes absent from the pinned table: {missing}")
        return (torch.tensor([self.table[n] for n in names], dtype=torch.float32) * 100.0).reshape(1, len(names), 3).to(device)

    def forward(self, x, pos):
        x = F.interpolate(x, size=round(x.shape[-1] * 200 / 120), mode="linear", align_corners=False)
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True, correction=1) + 1e-6)  # README: per channel and window
        tokens = self.body(x, pos.expand(len(x), -1, -1)).flatten(1, 2)
        pooled = (self.score(tokens).softmax(1) * tokens).sum(1)
        return self.task(pooled), self.motor(pooled)


def split_optimizer(model):
    head = list(model.score.parameters()) + list(model.task.parameters()) + list(model.motor.parameters())
    head_set = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_set]
    require(len(body) + len(head) == len(list(model.parameters())) and body, "optimizer groups do not partition")
    return torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
