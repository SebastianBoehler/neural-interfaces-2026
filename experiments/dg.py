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
"""

import torch
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


def averager(net):
    return AveragedModel(net)


def finish(avg, X, dev, batch=256):
    """Averaged net with batch-norm statistics recomputed on ``X``."""
    update_bn((X[b:b + batch].to(dev) for b in range(0, len(X), batch)), avg.module)
    return avg.module
