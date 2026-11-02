"""Multi-task co-training with the external MI corpora (AlignedNet ``cotrain``, 2 Nov).

Pretrain-then-fine-tune (R10-R16) moves the whole trunk to PROTEUS for 40 epochs, so
what it learned from hundreds of external participants can be overwritten. Here every
PROTEUS step also draws a balanced batch from the seven m7 corpora (each on its own
output block, as experiments/pretrain.py) and adds lam x its cross entropy. The net
has 3 + 19 outputs; PROTEUS uses the first 3 (training and prediction), so a packaged
member keeps only those rows of the final layer.
"""

import numpy as np
import torch
from torch import nn

M7 = "physionetmi,cho2017,lee2019_mi,dreyer2023,schirrmeister2017,shin2017a,shin2017b"
_cache = {}


class CoTrain:
    def __init__(self, lam, batch, dev, gen, rot=0.0, corpora=M7):
        if corpora not in _cache:  # once per process: the folds share it
            from pretrain import load_corpus
            Xs, Ys, blocks, off = [], [], [], 3
            for c in corpora.split(","):
                x, y, classes, _ = load_corpus(c, "ea")
                Xs.append(torch.from_numpy(x)), Ys.append(torch.from_numpy(y + off).long())
                blocks.append((off, off + len(classes)))
                off += len(classes)
            _cache[corpora] = Xs, Ys, blocks, off
        self.X, self.Y, self.blocks, self.n_out = _cache[corpora]
        self.lam, self.per, self.dev, self.gen, self.rot = lam, max(1, batch // len(self.X)), dev, gen, rot

    def loss(self, net):
        from dg import rotate
        xs, ys, lo, hi = [], [], [], []
        for X, Y, (a, b) in zip(self.X, self.Y, self.blocks):
            j = torch.randint(len(X), (self.per,), generator=self.gen)
            xs.append(X[j]), ys.append(Y[j])
            lo.append(torch.full((self.per,), a)), hi.append(torch.full((self.per,), b))
        xb, yb = torch.cat(xs).to(self.dev), torch.cat(ys).to(self.dev)
        if self.rot:
            xb = rotate(xb, self.gen, self.rot)
        out = net(xb)
        k = torch.arange(out.shape[1], device=self.dev)[None]
        own = (k >= torch.cat(lo).to(self.dev)[:, None]) & (k < torch.cat(hi).to(self.dev)[:, None])
        return self.lam * nn.functional.cross_entropy(out.masked_fill(~own, -1e4), yb)

