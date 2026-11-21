"""Fusing Shallow's temporal and spatial convs (both linear, nothing between them) into one conv1d with 41 input channels:
cost and feature difference against the signed forward, batch 64, per net kind.

python bench_team_b_fused.py ZIPS_DIR DATA OUT.json ZIP_NAME"""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

import check_hard_team_b as C
from bench_team_b_conv import clock


def fuse(body):
    w = body.tc.weight.double()[:, 0, 0, :]          # (f, k)
    s = body.sc.weight.double()[:, :, :, 0]            # (g, f, c)
    m = torch.einsum("gfc,fk->gck", s, w).float()
    const = torch.einsum("gfc,f->g", s, body.tc.bias.double()).float()
    return m, const


def fused_feats(net, x):
    b = net.body
    m, const = b.fused
    h = F.conv1d(b.bandpassed(x), m, const, padding=m.shape[-1] // 2)[:, :, None]
    h = b.bn(h)
    z = torch.log(b.pool(h * h).clamp_min(1e-6)).flatten(1)
    return net.bn(z)


def main(zips, data_path, out, name):
    data = C.D.load(Path(data_path))
    with TemporaryDirectory() as scratch:
        directory = Path(scratch) / name
        module = C.unpack(Path(zips) / f"{name}.zip", directory)
        model = C.load(module, directory, data, "cuda:0")
    team = model.team_b
    g = module._TEAM_B
    x = torch.from_numpy(data["X"][:64]).cuda()
    x = (x - x.mean(-1, keepdim=True)).float()
    res = dict(gpu=torch.cuda.get_device_name())
    with torch.no_grad():
        for kind, sl in (("shallow", slice(0, 3)), ("slow", slice(3, 6))):
            nets = team.nets[sl]
            for n in nets:
                n.body.fused = fuse(n.body)
                n.body.bandpassed = lambda x, body=n.body: g["bandpass"](x, body.mask)
            ref = [n.feats(x) for n in nets]
            new = [fused_feats(n, x) for n in nets]
            res[kind] = dict(max_abs_diff=max(float((a - b).abs().max()) for a, b in zip(ref, new)), orig_ms=clock(lambda: [n.feats(x) for n in nets]),
                             fused_ms=clock(lambda: [fused_feats(n, x) for n in nets]))
            print(kind, res[kind], flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
