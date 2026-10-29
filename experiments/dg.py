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

``mstyle`` = p (R18): MixStyle (Zhou et al. 2021). With probability p per
batch, the per-window feature statistics (mean and std over all but the
feature axis) after the first two convolutions are mixed with those of a
random other window, mostly another participant's. Hooks exist only while
training, so the exported net is unchanged.

``adv`` = lam (R18): participant-adversarial training (DANN, Ganin et al.
2016). A linear participant classifier reads the input of the final layer
through gradient reversal (scale lam), pushing the trunk towards features
that do not identify the participant.
"""

import torch
import torch.nn.functional as F
from torch import nn
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


class MixStyle:
    def __init__(self, net, gen, n=2, alpha=0.1):
        self.gen, self.on, self.beta = gen, False, torch.distributions.Beta(alpha, alpha)
        convs = [m for m in net.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))][:n]
        self.hooks = [m.register_forward_hook(self._hook) for m in convs]

    def _hook(self, module, inp, out):
        if not (self.on and module.training) or len(out) < 2:
            return out
        dims = tuple(range(2, out.dim()))
        mu, sig = out.mean(dims, keepdim=True), (out.var(dims, keepdim=True) + 1e-6).sqrt()
        mu, sig = mu.detach(), sig.detach()
        perm = torch.randperm(len(out), generator=self.gen).to(out.device)
        lam = self.beta.sample((len(out),) + (1,) * (out.dim() - 1)).to(out.device)
        mix_mu, mix_sig = lam * mu + (1 - lam) * mu[perm], lam * sig + (1 - lam) * sig[perm]
        return (out - mu) / sig * mix_sig + mix_mu

    def remove(self):
        for h in self.hooks:
            h.remove()


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Adversary:
    def __init__(self, net, n_groups, lam, lr, dev):
        self.n, self.lam, self.lr, self.dev, self.head = n_groups, lam, lr, dev, None
        self.hook = net.final_layer.register_forward_pre_hook(self._grab)

    def _grab(self, module, args):
        self.feat = args[0]

    def loss(self, g):
        f = _Reverse.apply(self.feat.flatten(1), self.lam)
        if self.head is None:
            self.head = nn.Linear(f.shape[1], self.n).to(self.dev)
            self.opt = torch.optim.AdamW(self.head.parameters(), lr=self.lr)
        return F.cross_entropy(self.head(f), g)

    def step(self):
        self.opt.step()
        self.opt.zero_grad()

    def remove(self):
        self.hook.remove()
