"""Lane C4: NeuroRVQ-EEG foundation-model encoder (braindecode/neurorvq-eeg-pretrained) as a frozen Lane C member.

Torch-only copy of braindecode's `NeuroRVQ` encoder (pinned braindecode commit 26ae8b4b, file models/neurorvq.py, which the
port card states equals the authors' module on random inputs); weights are the pinned port (HF revision 56ce74ce, safetensors
SHA-256 2e53d170...). CC BY-NC 4.0 (non-commercial) checkpoint and code. No `trust_remote_code`, no braindecode import.
The released checkpoint has no classification head; ours is `common.HeadMember`. Positions are channel slots (long, [1, C]).
"""
import json
import math
import struct
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import HeadMember, zscore
from oracle import require, sha

REVISION = "56ce74ce061428ec7829a648c29a462e8ca254ff"
HASHES = {"model.safetensors": "2e53d1702989909076331edcf144e3bde6a97b0fea0fa817b4c0b745a2c36cd8",
          "config.json": "09babe2d9661d022bebd1994109d591e7fd890ef38a8d122f7217aa47bdba428"}
CHANNELS = (
    "a1", "a2", "af3", "af4", "af7", "af8", "afz", "c1", "c2", "c3", "c4", "c5", "c6", "ccp1", "ccp2", "ccp3",
    "ccp4", "ccp5", "ccp6", "ccp7", "ccp8", "cfc1", "cfc2", "cfc3", "cfc4", "cfc5", "cfc6", "cfc7", "cfc8", "cp1",
    "cp2", "cp3", "cp4", "cp5", "cp6", "cpz", "cz", "eog", "f1", "f10", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
    "f9", "fc1", "fc2", "fc3", "fc4", "fc5", "fc6", "fcz", "fp1", "fp2", "fpz", "ft7", "ft8", "fz", "iz", "loc",
    "o1", "o2", "oz", "p08", "p1", "p10", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "po1", "po10", "po2",
    "po3", "po4", "po7", "po8", "po9", "poz", "pz", "roc", "sp1", "sp2", "t1", "t10", "t2", "t3", "t4", "t5", "t6",
    "t7", "t8", "t9", "tp10", "tp7", "tp8", "tp9",
)
DIM, HEADS, DEPTH, PATCH, MAX_PATCHES = 200, 10, 12, 200, 256


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, 3 * DIM, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(DIM)), nn.Parameter(torch.zeros(DIM))
        self.q_norm, self.k_norm = nn.LayerNorm(DIM // HEADS, eps=1e-6), nn.LayerNorm(DIM // HEADS, eps=1e-6)
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x):
        b, n, _ = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        q, k, v = (self.qkv(x) + bias.to(x.dtype)).reshape(b, n, 3, HEADS, -1).permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(self.q_norm(q).type_as(v), self.k_norm(k).type_as(v), v, scale=(DIM // HEADS) ** -0.5)
        return self.proj(a.transpose(1, 2).reshape(b, n, -1))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.attn = Attention()
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))
        self.gamma_1, self.gamma_2 = nn.Parameter(1e-5 * torch.ones(DIM)), nn.Parameter(1e-5 * torch.ones(DIM))

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        return x + self.gamma_2 * self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """Four temporal-conv scales; (kernel rows) as released; each branch -> [B, tokens, 200]."""
    K1, K2 = (21, 15, 9, 5), (9, 7, 5, 3)

    def __init__(self):
        super().__init__()
        for i, k in enumerate(self.K1, 1):
            setattr(self, f"conv1_{i}", nn.Conv2d(1, 8, (1, k), padding=(0, k // 2)))
            setattr(self, f"norm1_{i}", nn.GroupNorm(4, 8))
        for i, k in enumerate(self.K2, 1):
            setattr(self, f"conv2_{i}", nn.Conv2d(8, 8, (1, k), padding=(0, k // 2)))
            setattr(self, f"norm2_{i}", nn.GroupNorm(4, 8))

    def forward(self, x):  # [B, tokens, 200]
        b, n, _ = x.shape
        x = x.unsqueeze(1)
        out = []
        for i in range(1, 5):
            y = F.avg_pool2d(F.gelu(getattr(self, f"norm1_{i}")(getattr(self, f"conv1_{i}")(x))), (1, 2))
            y = F.avg_pool2d(F.gelu(getattr(self, f"norm2_{i}")(getattr(self, f"conv2_{i}")(y))), (1, 4))
            out.append(y.permute(0, 2, 3, 1).reshape(b, n, -1))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(len(CHANNELS) + 1, DIM))
        self.time_embed = nn.Parameter(torch.zeros(MAX_PATCHES, DIM))
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])

    def forward(self, x, slots):
        """[B, C, T] at 200 Hz (T a multiple of 200), slots [C] long -> [B, C * patches, 4 * 200] (scales concatenated)."""
        b, c, t = x.shape
        require(t % PATCH == 0 and t // PATCH <= MAX_PATCHES, "window is not whole 200-sample patches")
        p = t // PATCH
        branches = self.patch_embed(x.reshape(b, c * p, PATCH))
        spatial = self.pos_embed[F.pad(slots.repeat_interleave(p), (1, 0), value=0)].unsqueeze(0)
        temporal = self.time_embed[torch.arange(MAX_PATCHES - p, MAX_PATCHES, device=x.device).repeat(c)].unsqueeze(0)
        scales = torch.cat([torch.cat((self.cls_token.expand(b, -1, -1), u), 1) for u in branches])  # four scales as one batch
        scales = scales + spatial
        scales = torch.cat((scales[:, :1], scales[:, 1:] + temporal), 1)
        for block in self.blocks:
            scales = block(scales)
        return torch.cat(scales[:, 1:].chunk(4), dim=-1)

    def random_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias) if m.bias is not None else None
        for p in (self.cls_token, self.pos_embed, self.time_embed):
            nn.init.trunc_normal_(p, std=0.02)
        with torch.no_grad():
            for i, block in enumerate(self.blocks, 1):
                block.attn.proj.weight.div_(math.sqrt(2.0 * i))
                block.mlp[2].weight.div_(math.sqrt(2.0 * i))


def release_tensors(path):
    with path.open("rb") as fd:
        size, = struct.unpack("<Q", fd.read(8))
        header = json.loads(fd.read(size))
        data = memoryview(fd.read())
    out = {}
    for name, d in header.items():
        if name != "__metadata__":
            require(d["dtype"] == "F32", f"unsupported tensor type: {name}")
            a, b = d["data_offsets"]
            out[name] = torch.frombuffer(bytearray(data[a:b]), dtype=torch.float32).reshape(d["shape"])
    return out


class Member(HeadMember):
    def __init__(self, checkpoint, init):
        checkpoint = Path(checkpoint)
        require(init in ("pretrained", "random"), "init must be pretrained or random")
        for name, digest in HASHES.items():
            require(sha(checkpoint.parent / name) == digest, f"{name} differs from pinned revision {REVISION}")
        body = Encoder()
        if init == "pretrained":
            load = body.load_state_dict(release_tensors(checkpoint), strict=False)
            require(not load.missing_keys and not load.unexpected_keys, f"incomplete NeuroRVQ load: {load}")
        else:
            body.random_init()  # same architecture, same seed discipline as the fit; the control for "pretraining adds nothing"
        super().__init__(body, 4 * DIM, nn.Embedding(len(CHANNELS), 128))
        self.init = init

    def positions(self, names, device):
        missing = [n for n in names if n.lower() not in CHANNELS]
        require(not missing, f"electrodes absent from the NeuroRVQ vocabulary: {missing}")
        return torch.tensor([CHANNELS.index(n.lower()) for n in names], dtype=torch.long, device=device).reshape(1, -1)

    def tokens(self, x, pos):
        x = F.interpolate(x.float(), size=round(x.shape[-1] * 200 / 120), mode="linear", align_corners=False)
        return self.body(zscore(x), pos[0]), pos[0].repeat_interleave(x.shape[-1] // PATCH)
