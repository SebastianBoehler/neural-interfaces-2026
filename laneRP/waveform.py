"""Supervised single-visit fine-tuning of two actual cap250 waveform members."""
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneQ"))
from recog_lib import Gate

SEED = 20261120


class Fingerprint:
    def __init__(self, file, data):
        with np.load(file) as z:
            params = {"bio_" + k: z[k] for k in ("sd", "V", "B", "W", "S")}
        params.update(bio_mu=np.zeros_like(params["bio_sd"]), bio_gal=np.zeros((1, 64)), bio_m=[1],
                      bio_t0=10, bio_ramp=.05, bio_tau=.26)
        self.gate = Gate(params)
        self.proj = {k: self.gate.proj(v["spec"]) for k, v in data.items()}
        self.mean = {k: p[data[k]["day"] == 1].mean(0) for k, p in self.proj.items()}

    def route(self, names, probe):
        gallery = np.stack([self.mean[k] for k in names])
        center = gallery.mean(0)
        g, m = gallery - center, np.ones(len(gallery), int)
        prefix = (probe - center).cumsum(0)
        for t in range(10, len(probe)):
            sc = self.gate.score(g, m, prefix[t:t+1] / (t+1), t+1)[0]
            i = int(sc.argmax())
            if sc[i] > .26 and sc[i] - np.partition(sc, -2)[-2] > .04:
                return t, names[i]
        return -1, ""


def zero(module):
    with torch.no_grad():
        for name, v in module.named_buffers():
            if name.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
                v.detach_()
                v.zero_()


class Traces:
    def __init__(self, path):
        if path.is_dir():
            read = lambda name: (path / name).read_bytes()
            self.files = self.read(read)
        else:
            with zipfile.ZipFile(path) as f:
                self.files = self.read(f.read)

    @staticmethod
    def read(read):
        entries = {}
        with np.load(io.BytesIO(read("weights.npz"))) as z:
            indices = dict(zip(z["v6_slots"].astype(str), z["v6_ft_member"]))
            for name in ("fbraw", "fbl"):
                key = f"net{indices[name]}_"
                entries[name] = dict(blob=read(str(z[key+"sjit_file"])),
                                     **{k: z[key+k] for k in ("align", "scale", "G", "k", "shrink")})
        return entries


def aligned(x, p):
    if not p["align"]:
        return x / float(p["scale"])
    raw = x.double()
    delta = raw - raw.mean(-1, keepdim=True)
    cov = delta @ delta.transpose(1, 2) / raw.shape[-1]
    shrink = float(p["shrink"])
    cov = ((1 - shrink) * cov + shrink * cov.diagonal(dim1=1, dim2=2).mean(1)[:, None, None]
           * torch.eye(x.shape[1], device=x.device))
    n = torch.arange(1, len(x)+1, dtype=torch.float64, device=x.device)[:, None, None]
    k = float(p["k"])
    ref = (k * torch.as_tensor(p["G"], device=x.device) + cov.cumsum(0)) / (k + n)
    val, vec = torch.linalg.eigh(ref)
    w = (vec * val.clamp_min(1e-12).rsqrt()[:, None, :]) @ vec.transpose(1, 2)
    return (w @ raw).float()


class WaveExpert:
    def __init__(self, traces, x, y):
        self.members = []
        y = torch.as_tensor(y, dtype=torch.long, device="cuda")
        x = torch.as_tensor(x, dtype=torch.float32, device="cuda")
        class_weights = len(y) / (3 * y.bincount(minlength=3))
        assert class_weights.isfinite().all() and len(x) == len(y)
        for p in traces.files.values():
            model = torch.jit.load(io.BytesIO(p["blob"]), map_location="cuda").eval()
            params = [v for v in model.parameters() if v.requires_grad]
            params += [v for n, v in model.named_buffers() if n.rsplit(".", 1)[-1] in ("w", "b")]
            for v in params:
                v.requires_grad_(True)
            before = params[0].detach().clone()
            opt = torch.optim.Adam(params, lr=.0005)
            gen = torch.Generator(device="cuda").manual_seed(SEED)
            with torch.jit.optimized_execution(False):
                for _ in range(60):
                    ix = torch.randperm(len(x), device="cuda", generator=gen)[:48]
                    opt.zero_grad()
                    zero(model)
                    logits = model(aligned(x[ix], p))[:, :3]
                    loss = F.cross_entropy(logits, y[ix], weight=class_weights)
                    loss.backward()
                    opt.step()
            if not loss.isfinite() or torch.equal(params[0].detach(), before):
                raise ValueError("inactive waveform fine-tuning")
            for v in params:
                v.requires_grad_(False)
            self.members.append((model, p))
            zero(model)

    @torch.no_grad()
    def predict(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device="cuda")
        all_probs = []
        for m, p in self.members:
            zero(m)
            with torch.jit.optimized_execution(False):
                y = m(aligned(x, p))[:, :3].softmax(1).cpu().numpy()
            zero(m)
            all_probs.append(y)
        probs = np.stack(all_probs, axis=1)
        if not np.isfinite(probs).all():
            raise ValueError("nonfinite waveform posterior")
        return probs
