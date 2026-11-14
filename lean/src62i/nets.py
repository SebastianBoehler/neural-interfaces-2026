"""Network members. Every net maps an input batch (B, C, T) to float64 class posteriors (B, 3)."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .inputs import INPUT_KEYS, InputStream, input_key

# SessionBN traces whose FIR filter bank is identical (checked on load): one bank output serves a whole input group
FBC_TRACES = frozenset(f"{fam}_s{seed}.pt" for fam in ("sbn_k2", "sbn_k5", "sbn_fbcnet_raw_k20") for seed in range(6))
FBL_TRACES = frozenset(f"sbn_fblight_ea_rot05_k5_s{seed}.pt" for seed in range(6))
SBN_FIR_SHA256 = "607a44ee25576803092c03d77feb67411fe7a2266475fc730ae7ae28c81506c6"


class Net:
    """Interface: ``proba(X, cache)``; ``cache`` is shared by the nets of one input group for one window."""
    dtype = torch.float32

    def reset_state(self):
        pass


class Shallow(Net):
    """ShallowConvNet exported as plain tensors (temporal and spatial filters merged into one kernel)."""

    def __init__(self, p, dev, root):
        t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in p.items() if k not in INPUT_KEYS}
        self.t = t
        self.weight = (t["time_w"] * t["spat_w"].permute(1, 0, 2, 3)).sum(0).unsqueeze(1)
        self.bias = t["spat_w"].squeeze().sum(-1).mm(t["time_b"].unsqueeze(-1)).squeeze()

    def proba(self, X, cache):
        t = self.t
        h = F.conv2d(X.unsqueeze(1).transpose(2, 3), self.weight, self.bias)
        h = F.batch_norm(h, t["bn_mean"], t["bn_var"], t["bn_w"], t["bn_b"], False, 0.0, 1e-5)
        h = torch.log(torch.clamp(F.avg_pool2d(h * h, (75, 1), (15, 1)), min=1e-6))
        return torch.softmax(F.conv2d(h, t["cls_w"], t["cls_b"])[:, :, 0, 0].double(), 1)


def fft_bank(x, fir, spectra, fir_spectra):
    """"Same"-padded FIR filtering by FFT, (B, C, T) -> (B, K, C, T). ``spectra`` caches the input's spectra per FFT
    length (shared within an input group), ``fir_spectra`` the filters' (kept by the net)."""
    n, out = x.shape[-1], []
    for i, f in enumerate(fir):
        m = n + f.shape[-1] - 1
        if m not in spectra:
            spectra[m] = torch.fft.rfft(x, m)
        if (i, m) not in fir_spectra:
            fir_spectra[i, m] = torch.fft.rfft(f, m)
        s = (m - n) // 2
        out.append(torch.fft.irfft(spectra[m] * fir_spectra[i, m], m)[..., s:s + n])
    return torch.stack(out, 1)


class FBC(Net):
    """FBCNet exported as plain float64 tensors: FIR bank, grouped spatial filters, batch norm, SiLU, log-variance
    over ``stride`` segments, linear head."""
    dtype = torch.float64

    def __init__(self, p, dev, root):
        self.t = {k: torch.from_numpy(np.ascontiguousarray(v)).double().to(dev)
                  for k, v in p.items() if k.startswith("fbc_")}
        n = int(p["fbc_n_bands"])
        self.fir = [self.t[f"fbc_fir{i}"] for i in range(n)]
        self.bank_key = b"".join(np.asarray(p[f"fbc_fir{i}"], np.float64).tobytes() for i in range(n))
        self.fir_spectra, self.stride = {}, int(p["fbc_stride"])

    def proba(self, X, cache):
        t = self.t
        if self.bank_key not in cache:
            cache[self.bank_key] = fft_bank(X, self.fir, cache.setdefault("spectra", {}), self.fir_spectra)
        Y = cache[self.bank_key]
        K, C = Y.shape[1], Y.shape[2]
        Z = torch.einsum("kfc,bkct->bkft", t["fbc_W"].view(K, -1, C), Y)
        Z = Z.reshape(X.shape[0], -1, X.shape[-1]) + t["fbc_bw"][None, :, None]
        Z = (Z - t["fbc_rm"][None, :, None]) / torch.sqrt(t["fbc_rv"][None, :, None] + t["fbc_eps"])
        Z = F.silu(Z * t["fbc_g"][None, :, None] + t["fbc_be"][None, :, None])
        V = Z.view(Z.shape[0], Z.shape[1], self.stride, Z.shape[-1] // self.stride).var(dim=3)
        V = torch.log(V.clamp(1e-6, 1e6)).flatten(1)
        return torch.softmax(V @ t["fbc_L"].T + t["fbc_lb"], 1)


class Jit(Net):
    """Stateless TorchScript member (the external teacher): input -> logits. Session copies share the module."""
    file_key = "jit_file"

    def __init__(self, p, dev, root):
        self.m = torch.jit.load(str(Path(root) / str(p[self.file_key])), map_location=dev).eval()

    def __deepcopy__(self, memo):
        return self

    def proba(self, X, cache):
        return torch.softmax(self.m(X).double(), 1)


class SJit(Jit):
    """SessionBN TorchScript member: its batch norms use the session's causal statistics, held in buffers named
    s1 / s2 / n. Every session copy owns its module."""
    file_key = "sjit_file"

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        new.m = copy.deepcopy(self.m, memo)
        return new

    def reset_state(self):
        for name, b in self.m.named_buffers():
            if name.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
                b.zero_()


class SharedFIRSJit(SJit):
    """SessionBN FBCNet / FBLightConvNet trace run from a filter-bank output shared within its input group."""

    def __init__(self, p, dev, root):
        super().__init__(p, dev, root)
        name = str(p["sjit_file"])
        self.fblight = name in FBL_TRACES
        fir = [f.detach().cpu().numpy() for _, f in self.m.spectral_filtering.fir_list.named_parameters()]
        digest = hashlib.sha256(b"".join(f.astype("<f4", copy=False).tobytes() for f in fir)).hexdigest()
        if self.m.original_name != ("FBLightConvNet" if self.fblight else "FBCNet") or digest != SBN_FIR_SHA256:
            raise ValueError(f"{name}: not the fingerprinted SessionBN trace this code path was written for")

    def proba(self, X, cache):
        if "sbn_bank" not in cache:
            cache["sbn_bank"] = self.m.spectral_filtering(X)
        Y = cache["sbn_bank"]
        h = self.m.spatial_conv(Y)  # the traces' forward after spectral_filtering
        if self.fblight:
            h = h[..., :250].reshape(Y.shape[0], 32, -1, 250)
            h = self.m.attn_conv(self.m.temporal_layer(h))
        else:
            h = self.m.temporal_layer(h.view(h.shape[0], h.shape[1], 4, 120))
        return torch.softmax(self.m.final_layer(self.m.flatten_layer(h)).double(), 1)


def build_net(p, dev, root):
    """The member class for one exported net; exports this package does not use fail loudly."""
    if "sjit_file" in p:
        shared = str(p["sjit_file"]) in FBC_TRACES | FBL_TRACES
        return (SharedFIRSJit if shared else SJit)(p, dev, root)
    if "jit_file" in p:
        return Jit(p, dev, root)
    if "fbc_n_bands" in p and "fbl_win" not in p:
        return FBC(p, dev, root)
    if "time_w" in p:
        return Shallow(p, dev, root)
    raise ValueError(f"unsupported net export with keys {sorted(p)}")


class NetBank:
    """All net members with non-zero weight, grouped by input stream. Calling it with one window returns
    [(member index, posterior (3,))] in the order the mix adds them: groups by first appearance among all exported
    nets, members by index within a group."""

    def __init__(self, nets, live, dev, root):
        groups = {}
        for i, p in enumerate(nets):
            members = groups.setdefault(input_key(p), (InputStream(p), []))[1]
            if live[i]:
                members.append((i, build_net(p, dev, root)))
        self.groups, self.dev = [g for g in groups.values() if g[1]], dev

    def reset_state(self):
        for stream, members in self.groups:
            stream.reset_state()
            for _, net in members:
                net.reset_state()

    @torch.no_grad()
    def __call__(self, x):
        ids, probs, self.last_inputs = [], [], {}
        for stream, members in self.groups:
            h = stream(x)
            self.last_inputs.update({i: h for i, _ in members})  # v6 fine-tuning buffers its members' inputs
            H = torch.from_numpy(h[None])
            inputs, cache = {}, {}
            for i, net in members:
                if net.dtype not in inputs:
                    inputs[net.dtype] = H.to(dtype=net.dtype).to(self.dev)
                ids.append(i)
                probs.append(net.proba(inputs[net.dtype], cache))
        return list(zip(ids, torch.stack(probs).cpu().numpy()[:, 0]))
