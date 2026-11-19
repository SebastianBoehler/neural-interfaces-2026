"""Pratibha-2 / MANAS-2 encoder, Apache-2.0. Adapted from MannasAI/pratibha-2 at 863edc940d82a02744776a7820641bf1bc32310b via laneC3/pratibha.py."""
import math
import torch
from torch import nn
from torch.nn import functional as F

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
        if eeg.ndim != 3 or eeg.shape[-1] < 200 or tuple(positions.shape) != (*eeg.shape[:2], 3):
            raise ValueError("Invalid Pratibha input shape")
        batch, channels, _ = eeg.shape
        tokens = self.patch_embed(eeg)
        patches = tokens.shape[2]
        spatial = positions.unsqueeze(2).expand(-1, -1, patches, -1)
        time = torch.arange(patches, device=eeg.device, dtype=torch.float32).view(1, 1, patches, 1)
        coords = torch.cat([spatial, time.expand(batch, channels, -1, -1)], dim=-1).reshape(batch, channels * patches, 4)
        encoded = self.encoder(tokens.flatten(1, 2) + self.pos_enc(coords))
        return encoded.reshape(batch, channels, patches, 512).permute(0, 2, 1, 3).contiguous()


