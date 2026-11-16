"""Source-only model selection primitives for registered strong heads."""

import numpy as np
import torch
from torch.nn import functional as F

from heads import RandomMap, Structured, fit_scale, ridge_fit, ridge_predict


def _labels(y, count):
    y = np.asarray(y, dtype=np.int64)
    if y.shape != (count,) or not np.isin(y, (0, 1, 2)).all():
        raise ValueError('invalid labels')
    return y


def _loss(probabilities, labels):
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise ValueError('invalid selection probabilities')
    return float(-np.log(probabilities[np.arange(len(labels)), labels]).mean())


def _training(x, y, nonlinear, seed, epochs, batch, device):
    labels = torch.as_tensor(y, dtype=torch.long, device=device)
    torch.manual_seed(seed)
    model = Structured(nonlinear).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    sampler = torch.Generator(device='cpu').manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(x), generator=sampler).split(batch):
            optimizer.zero_grad(set_to_none=True)
            ids = indices.to(device)
            loss = F.cross_entropy(model(x[ids]), labels[ids])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite structured training loss')
            loss.backward()
            optimizer.step()
        yield epoch, model


def fit_structured(x, y, validation, validation_y, nonlinear, seed, epochs=(10, 20, 30),
                   batch=128, device='cpu'):
    x = np.asarray(x, dtype=np.float32)
    validation = np.asarray(validation, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 44239 or validation.ndim != 2 or validation.shape[1] != 44239:
        raise ValueError('structured inputs must have 44239 features')
    y = _labels(y, len(x))
    validation_y = _labels(validation_y, len(validation))
    if len(np.unique(y)) != 3 or not len(validation) or not epochs or min(epochs) <= 0:
        raise ValueError('structured selection requires three source classes and validation')
    mean, scale = fit_scale(x)
    train_x = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    valid_x = torch.as_tensor((validation - mean) / scale, dtype=torch.float32, device=device)
    best, best_state, losses = None, None, {}
    for epoch, model in _training(train_x, y, nonlinear, seed, max(epochs), batch, device):
        if epoch in epochs:
            model.eval()
            with torch.no_grad():
                probabilities = model(valid_x).softmax(dim=1).cpu().numpy()
            losses[epoch] = _loss(probabilities, validation_y)
            if best is None or losses[epoch] < losses[best]:
                best = epoch
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, (mean, scale), best, losses


def fit_ridge_grid(x, y, validation, validation_y, arm, seed, width=512):
    x = np.asarray(x, dtype=np.float32)
    validation = np.asarray(validation, dtype=np.float32)
    y = _labels(y, len(x))
    validation_y = _labels(validation_y, len(validation))
    if len(np.unique(y)) != 3 or not len(validation):
        raise ValueError('ridge selection requires three source classes and validation')
    mean, scale = fit_scale(x)
    source = np.asarray((x - mean) / scale, dtype=np.float32)
    valid = np.asarray((validation - mean) / scale, dtype=np.float32)
    mapper = RandomMap.fit(source, arm, seed, width=width)
    z, z_valid = mapper.transform(source), mapper.transform(valid)
    losses, choices = {}, []
    for strength in (1e-4, 1e-3):
        weights = ridge_fit(z, y, strength)
        for temperature in (1, 4):
            loss = _loss(ridge_predict(z_valid, weights, temperature), validation_y)
            losses[f'{strength}:{temperature}'] = loss
            choices.append((loss, strength, temperature, weights))
    _, strength, temperature, weights = min(choices, key=lambda entry: entry[:3])
    return {'mean': mean, 'scale': scale, 'mapper': mapper, 'weights': weights,
            'strength': strength, 'temperature': temperature, 'validation': losses}


def refit_structured(x, y, nonlinear, seed, epochs, batch=128, device='cpu'):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 44239 or epochs <= 0 or batch <= 0:
        raise ValueError('invalid structured refit arguments')
    y = _labels(y, len(x))
    if len(np.unique(y)) != 3:
        raise ValueError('structured refit requires three source classes')
    mean, scale = fit_scale(x)
    train_x = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    for _, model in _training(train_x, y, nonlinear, seed, epochs, batch, device):
        pass
    return model.eval(), (mean, scale)


def refit_ridge(x, y, arm, seed, strength, temperature, width=512):
    x = np.asarray(x, dtype=np.float32)
    y = _labels(y, len(x))
    if len(np.unique(y)) != 3 or strength not in (1e-4, 1e-3) or temperature not in (1, 4):
        raise ValueError('invalid ridge refit selection or source labels')
    mean, scale = fit_scale(x)
    source = np.asarray((x - mean) / scale, dtype=np.float32)
    mapper = RandomMap.fit(source, arm, seed, width=width)
    weights = ridge_fit(mapper.transform(source), y, strength)
    return {'mean': mean, 'scale': scale, 'mapper': mapper, 'weights': weights,
            'strength': strength, 'temperature': temperature}
