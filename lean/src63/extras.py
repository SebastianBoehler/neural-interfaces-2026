"""v6 members for unrecognised sessions: the eye-net view and the rest-class member (docs/V6_DESIGN.md)."""
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter

from .inputs import CH_NAMES, InputStream

EYE_SETS = {"h1a": ("F5", "AF3", "F3"), "h1b": ("F6", "AF4", "F4"), "h2a": ("FC5", "C5"), "h2b": ("FC6", "C6"),
            "vt": ("AF3", "AF4"), "v1b": ("P1", "P2", "Pz"), "v2b": ("C1", "C2", "Cz")}


class EyeNet:
    """Seven derived ocular traces per window (two horizontal pairs, two vertical differences, magnitude, two
    velocities; studies/12nov/ocunet.py traces), each centred and scaled by the session's running statistics up to and
    including the window and clipped at +-8; the eye nets' mean posterior gives three log values (clipped at 1e-3)
    that are appended to SessionEM's view."""

    def __init__(self, z, dev, root):
        self.nets = [torch.jit.load(str(Path(root) / str(f)), map_location=dev).eval() for f in z["v6_eye_files"]]
        ix = {c: i for i, c in enumerate(CH_NAMES)}
        self.idx = {k: [ix[c] for c in cs] for k, cs in EYE_SETS.items()}
        self.dev = dev
        self.reset_state()

    def __deepcopy__(self, memo):  # the nets are stateless: session copies share them
        new = copy.copy(self)
        new.reset_state()
        return new

    def reset_state(self):
        self.n, self.s1, self.s2 = 0, 0.0, 0.0

    def traces(self, x):
        X = x.astype(np.float64)
        sel = {k: X[i].mean(0) for k, i in self.idx.items()}
        sg = lambda a: savgol_filter(a, 25, 3, axis=-1)  # noqa: E731
        h1, h2 = sg(sel["h1a"] - sel["h1b"]), sg(sel["h2a"] - sel["h2b"])
        v1, v2 = sg(sel["vt"] - sel["v1b"]), sg(sel["vt"] - sel["v2b"])
        T = np.stack([h1, h2, v1, v2, np.sqrt(h1 ** 2 + v1 ** 2), np.gradient(h1, axis=-1), np.gradient(v1, axis=-1)])
        self.n += 1
        self.s1, self.s2 = self.s1 + T.mean(-1), self.s2 + (T ** 2).mean(-1)
        m1, m2 = self.s1 / self.n, self.s2 / self.n
        return np.clip((T - m1[:, None]) / np.sqrt(np.maximum(m2 - m1 ** 2, 1e-12))[:, None], -8, 8).astype(np.float32)

    @torch.no_grad()
    def step(self, x):
        T = torch.from_numpy(self.traces(x)[None]).to(self.dev)
        p = torch.stack([torch.softmax(m(T).double(), 1)[0] for m in self.nets]).mean(0).cpu().numpy()
        return np.log(np.clip(p, 1e-3, 1))


class RestMember:
    """The rest-class SessionBN shallow nets (studies/11nov/rest4/rest4.py, sh r1): four outputs, used three-way
    (softmax over the first three), several seeds averaged, on their own causal input stream."""

    def __init__(self, z, dev, root):
        self.stream = InputStream({k: z[f"v6_rest_{k}"] for k in ("align", "k", "G", "scale", "shrink")})
        self.nets = [torch.jit.load(str(Path(root) / str(f)), map_location=dev).eval() for f in z["v6_rest_files"]]
        self.dev = dev

    def __deepcopy__(self, memo):  # StreamBN state lives in the modules: every session copy owns its modules
        new = copy.copy(self)
        new.stream, new.nets = copy.deepcopy(self.stream, memo), [copy.deepcopy(m, memo) for m in self.nets]
        return new

    def reset_state(self):
        self.stream.reset_state()
        for m in self.nets:
            for name, b in m.named_buffers():
                if name.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
                    b.zero_()

    @torch.no_grad()
    def step(self, x):
        H = torch.from_numpy(self.stream(x)[None]).to(self.dev)
        return torch.stack([torch.softmax(m(H)[:, :3].double(), 1)[0] for m in self.nets]).mean(0).cpu().numpy()
