"""Domain-generalisation training options for AlignedNet (R11).

``swa`` = f: average the weights of every step from fraction f of training on
(stochastic weight averaging along the annealed trajectory, as in SWAD, Cha
et al. 2021), then recompute batch-norm statistics on the training windows.
Flat minima are expected to transfer better to unseen participants. The
averaged net has the trained architecture, so packaging is unchanged.

``cmix`` = a: same-class mixing. Every window is blended with a random window
of its own class in the batch (weight lambda ~ Beta(a, a), the larger part
kept) and rescaled to preserve power. Most partners come from another
participant, so this interpolates between participants.

``rot`` = e (R14): every window is rotated by expm(A), A skew-symmetric with
N(0, e) entries. After alignment, participants differ mainly by a rotation
that only labels could undo (Riemannian Procrustes); this trains invariance
to it.

``fsc`` = s (R14): each batch is time-scaled by a factor in [1 - s, 1 + s]
(spectral peaks shift by the same factor; individual alpha frequency varies
by about +-15% across people), then cropped or circularly padded back.
"""

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, update_bn


def class_mix(xb, yb, gen, alpha):
    perm = torch.arange(len(yb))
    yc = yb.cpu()
    for c in yc.unique():
        i = torch.nonzero(yc == c).flatten()
        perm[i] = i[torch.randperm(len(i), generator=gen)]
    lam = torch.distributions.Beta(alpha, alpha).sample((len(xb), 1, 1))
    lam = torch.maximum(lam, 1 - lam).to(xb.device)
    mixed = lam * xb + (1 - lam) * xb[perm.to(xb.device)]
    return mixed / torch.sqrt(lam ** 2 + (1 - lam) ** 2)


def rotate(xb, gen, eps):
    C = xb.shape[1]
    A = eps * torch.randn(len(xb), C, C, generator=gen)
    Q = torch.linalg.matrix_exp(A - A.transpose(1, 2))
    return Q.to(xb.device) @ xb


def freq_scale(xb, gen, s):
    T = xb.shape[-1]
    f = 1 + s * (2 * float(torch.rand(1, generator=gen)) - 1)
    y = F.interpolate(xb, size=max(2, round(T / f)), mode="linear", align_corners=False)
    if y.shape[-1] >= T:
        k = int(torch.randint(0, y.shape[-1] - T + 1, (1,), generator=gen))
        return y[..., k:k + T]
    return F.pad(y, (0, T - y.shape[-1]), mode="circular")


def averager(net):
    return AveragedModel(net)


def finish(avg, X, dev, batch=256):
    """Averaged net with batch-norm statistics recomputed on ``X``."""
    update_bn((X[b:b + batch].to(dev) for b in range(0, len(X), batch)), avg.module)
    return avg.module
