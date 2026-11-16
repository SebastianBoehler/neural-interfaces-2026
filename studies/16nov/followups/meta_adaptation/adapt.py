"""§307 bounded, differentiable previous-prefix head adaptation."""
from collections import deque
import torch

BLOCKS, WIDTH, SLOW = 51, 861, 328
MUTABLE, FEATURES = BLOCKS * WIDTH, BLOCKS * WIDTH + SLOW


def loss(head, anchor, bias, x):
    p = torch.softmax(x @ head.T + bias, dim=1)
    logp = torch.log_softmax(x @ head.T + bias, dim=1)
    mean = p.mean(dim=0)
    entropy = -(p * logp).sum(dim=1).mean()
    diversity = -(mean * mean.log()).sum()
    return entropy - .1 * diversity + .1 * ((head[:, :MUTABLE] - anchor[:, :MUTABLE]) ** 2).sum() / (anchor[:, :MUTABLE] ** 2).sum()


def update(anchor, initial, bias, x, multiplier, create_graph=False):
    if anchor.shape != (3, FEATURES) or initial.shape != anchor.shape or bias.shape != (3,):
        raise ValueError('incorrect core head dimensions')
    if x.ndim != 2 or x.shape[1] != FEATURES or not 1 <= x.shape[0] <= 64:
        raise ValueError('support must contain 1–64 causal features')
    if multiplier.shape not in (torch.Size([]), torch.Size([BLOCKS])):
        raise ValueError('expected one or 51 multipliers')
    if not all(torch.isfinite(v).all().item() for v in (anchor, initial, bias, x, multiplier)):
        raise ValueError('nonfinite adaptation input')
    if (multiplier.abs() > 1).any().item():
        raise ValueError('multiplier outside registered bounds')
    norm = torch.linalg.vector_norm(anchor[:, :MUTABLE])
    if norm.item() <= 0: raise ValueError('zero population head norm')
    blocks = multiplier if multiplier.ndim == 0 else multiplier.repeat_interleave(WIDTH).unsqueeze(0)
    theta = initial
    max_grad, max_move = 0., 0.
    for _ in range(3):
        if not theta.requires_grad: theta = theta.detach().requires_grad_(True)
        objective = loss(theta, anchor, bias, x)
        grad = torch.autograd.grad(objective, theta, create_graph=create_graph)[0][:, :MUTABLE]
        grad_norm = torch.linalg.vector_norm(grad)
        if not torch.isfinite(grad_norm).item() or grad_norm.item() <= 0:
            raise ValueError('zero or nonfinite head gradient')
        step = .01 * norm * grad / grad_norm * blocks
        proposal = theta[:, :MUTABLE] - step
        diff = proposal - anchor[:, :MUTABLE]
        distance = torch.linalg.vector_norm(diff)
        cap = .03 * norm
        diff = diff * torch.clamp(cap / distance.clamp_min(torch.finfo(diff.dtype).tiny), max=1.)
        theta = torch.cat((anchor[:, :MUTABLE] + diff, anchor[:, MUTABLE:]), dim=1)
        max_grad = max(max_grad, float(grad_norm.detach()))
        max_move = max(max_move, float(torch.linalg.vector_norm(diff).detach() / norm.detach()))
    if not torch.isfinite(theta).all().item() or max_move > .03000001:
        raise ValueError('invalid bounded adaptation')
    return theta, {'gradient_max': max_grad, 'delta_ratio_max': max_move}


class Session:
    def __init__(self, anchor, bias, multiplier, enabled=True):
        self.anchor, self.bias, self.multiplier, self.enabled = anchor, bias, multiplier, enabled
        self.reset()

    def reset(self):
        self.head = self.anchor.detach().clone()
        self.prefix = deque(maxlen=64)
        self.count = 0
        self.updates = 0

    def step(self, x):
        if x.shape != (FEATURES,) or not torch.isfinite(x).all().item(): raise ValueError('invalid current feature')
        t = self.count
        if self.enabled and t >= 40 and t % 20 == 0:
            support = torch.stack(tuple(self.prefix))
            self.head, _ = update(self.anchor, self.head, self.bias, support, self.multiplier)
            self.head = self.head.detach(); self.updates += 1
        p = torch.softmax(x @ self.head.T + self.bias, dim=0)
        self.prefix.append(x.detach().clone())
        self.count += 1
        return p.detach()
