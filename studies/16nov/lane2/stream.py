"""Session-local, population-anchored pseudo-label adaptation of the last layer."""
import copy
import numpy as np
import torch
from torch.nn import functional as F

START, EVERY, STEPS = 20, 20, 3
CONF, LR, ANCHOR, MAX_DELTA = 0.75, 0.01, 0.1, 0.1


def predict_session(population, x, seed=0, update=True):
    """Rows in delivery order. Each call owns and resets its only mutable state."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=next(population.parameters()).device)
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError('features must be a finite matrix')
    pop = population.eval(); model = copy.deepcopy(pop).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.out.parameters():
        p.requires_grad_(True)
    base = [p.detach().clone() for p in model.out.parameters()]
    gen = torch.Generator(device=x.device).manual_seed(seed)
    static, result, counts = [], [], []
    for t in range(len(x)):
        if update and t >= START and t % EVERY == 0:
            # Only rows strictly before t determine the current update.
            prefix = x[:t]
            with torch.no_grad():
                pp = pop(prefix).softmax(1)
                confidence, label = pp.max(1)
                selected = confidence >= CONF
            n = int(selected.sum().item()); counts.append(n)
            if n:
                chosen, target = prefix[selected], label[selected]
                for _ in range(STEPS):
                    mask = (torch.rand(chosen.shape, generator=gen, device=x.device) >= 0.1).to(chosen.dtype)
                    penalty = sum(((p - b) ** 2).sum() for p, b in zip(model.out.parameters(), base))
                    loss = F.cross_entropy(model(chosen * mask), target) + ANCHOR * penalty
                    grads = torch.autograd.grad(loss, tuple(model.out.parameters()))
                    with torch.no_grad():
                        for p, grad in zip(model.out.parameters(), grads):
                            p.sub_(LR * grad)
                        norm = torch.sqrt(sum(((p - b) ** 2).sum() for p, b in zip(model.out.parameters(), base)))
                        factor = min(1.0, MAX_DELTA / max(float(norm), 1e-12))
                        for p, b in zip(model.out.parameters(), base):
                            p.copy_(b + factor * (p - b))
        with torch.no_grad():
            static.append(pop(x[t:t + 1]).softmax(1).cpu().numpy()[0])
            result.append(model(x[t:t + 1]).softmax(1).cpu().numpy()[0])
    delta = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(model.out.parameters(), base))))
    if delta > MAX_DELTA + 1e-6:
        raise AssertionError('anchor trust region exceeded')
    return np.asarray(static), np.asarray(result), {'selected': counts, 'delta': delta}
