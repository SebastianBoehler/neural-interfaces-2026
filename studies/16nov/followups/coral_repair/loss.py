"""Align only same-class covariances across training people; leave means free."""
import torch


def conditional(features, person, label):
    f = features.flatten(1)
    if (len(f) != len(person) or len(f) != len(label) or f.shape[1] == 0 or
            not torch.isfinite(f).all() or label.ndim != 1 or person.ndim != 1 or
            torch.any((label < 0) | (label > 2))):
        raise ValueError('invalid conditional CORAL batch')
    d = f.shape[1]; differences, counts = [], []
    for cls in range(3):
        covariances = []
        for identity in torch.unique(person):
            rows = f[(person == identity) & (label == cls)]
            if len(rows) >= 2:
                rows = rows - rows.mean(0)
                covariances.append(rows.T @ rows / (len(rows) - 1))
        pairs = [((a - b) ** 2).sum() for i, a in enumerate(covariances)
                 for b in covariances[i + 1:]]
        counts.append(len(pairs)); differences.extend(pairs)
    if not differences:
        return f.sum() * 0, 0, counts
    return sum(differences) / (4 * d * d * len(differences)), len(differences), counts


def training_penalty(features, person, label, stats, original, mode):
    """Track activity without reading held labels or changing the batch sampler."""
    if mode == 'unconditional':
        return original(features, person)
    if mode != 'conditional':
        raise ValueError('unknown registered CORAL mode')
    penalty, pairs, by_class = conditional(features, person, label)
    stats['batches'] += 1
    stats['zero_pair_batches'] += pairs == 0
    stats['class_pairs'] = [a + b for a, b in zip(stats['class_pairs'], by_class)]
    if pairs and penalty.detach().item() > 0 and stats['gradient_norm'] == 0:
        gradient = torch.autograd.grad(penalty, features, retain_graph=True)[0]
        norm = float(gradient.detach().norm())
        if not torch.isfinite(gradient).all() or norm <= 0:
            raise ValueError('conditional penalty has no finite gradient')
        stats['gradient_norm'] = norm
    return penalty, pairs
