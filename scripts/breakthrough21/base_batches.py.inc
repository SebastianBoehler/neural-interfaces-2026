class BatchedFBC:
    """Apply one spatial map and readout for same-bank, same-shape frozen FBC members."""

    def __init__(self, nets):
        self.first = nets[0]
        self.count = len(nets)
        self.bands = len(self.first.fir)
        self.stride = self.first.stride
        self.weights = torch.cat([m.t["fbc_W"].view(self.bands, -1, 41) for m in nets], dim=1)
        self.add = torch.stack([m.t["fbc_bw"] for m in nets])[None, ..., None]
        self.mean = torch.stack([m.t["fbc_rm"] for m in nets])[None, ..., None]
        self.sd = torch.stack([torch.sqrt(m.t["fbc_rv"] + m.t["fbc_eps"]) for m in nets])[None, ..., None]
        self.gain = torch.stack([m.t["fbc_g"] for m in nets])[None, ..., None]
        self.offset = torch.stack([m.t["fbc_be"] for m in nets])[None, ..., None]
        self.head = torch.stack([m.t["fbc_L"] for m in nets])
        self.bias = torch.stack([m.t["fbc_lb"] for m in nets])

    def proba(self, x, cache):
        m = self.first
        if m.bank_key not in cache:
            cache[m.bank_key] = fft_bank(x, m.fir, cache.setdefault("spectra", {}), m.fir_spectra)
        y = cache[m.bank_key]
        b, t = x.shape[0], x.shape[-1]
        z = torch.einsum("kfc,bkct->bkft", self.weights, y)
        z = z.reshape(b, self.bands, self.count, -1, t).transpose(1, 2).reshape(b, self.count, -1, t)
        z = (z + self.add - self.mean) / self.sd
        z = F.silu(z * self.gain + self.offset)
        v = z.view(b, self.count, z.shape[2], self.stride, t // self.stride).var(dim=4)
        v = torch.log(v.clamp(1e-6, 1e6)).flatten(2)
        q = torch.einsum("bmd,mkd->bmk", v, self.head) + self.bias
        return torch.softmax(q, dim=-1).unbind(1)


class BatchedShallow:
    """Reuse the signed merged kernels in a single convolution over each input group."""

    def __init__(self, nets):
        self.count = len(nets)
        self.weight = torch.cat([m.weight for m in nets])
        self.bias = torch.cat([m.bias for m in nets])
        self.bn = [torch.cat([m.t[name] for m in nets]) for name in ("bn_mean", "bn_var", "bn_w", "bn_b")]
        self.read = torch.cat([m.t["cls_w"] for m in nets])
        self.read_bias = torch.cat([m.t["cls_b"] for m in nets])

    def proba(self, x, _cache):
        h = F.conv2d(x.unsqueeze(1).transpose(2, 3), self.weight, self.bias)
        h = F.batch_norm(h, *self.bn, False, 0.0, 1e-5)
        h = torch.log(torch.clamp(F.avg_pool2d(h * h, (75, 1), (15, 1)), min=1e-6))
        z = F.conv2d(h, self.read, self.read_bias, groups=self.count)[:, :, 0, 0].double()
        return torch.softmax(z.reshape(x.shape[0], self.count, 3), -1).unbind(1)


def build_frozen_batches(groups):
    result = {}
    for _, members in groups:
        eligible = {}
        for i, net in members:
            if type(net) is FBC:
                kind = (BatchedFBC, net.bank_key, net.stride, tuple(net.t["fbc_W"].shape))
            elif type(net) is Shallow:
                kind = (BatchedShallow, tuple(net.weight.shape), tuple(net.t["cls_w"].shape))
            else:
                continue
            eligible.setdefault(kind, []).append((i, net))
        for (builder, *_), group in eligible.items():
            if len(group) > 1:
                ids, nets = zip(*group, strict=True)
                batch = builder(nets)
                for i in ids:
                    result[i] = (ids, batch)
    return result
