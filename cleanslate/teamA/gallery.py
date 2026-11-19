"""Fit calibrated static references and one personal head per enrolled person."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from data import align_trials, prior
from eeg import center, covariance, fingerprint


@torch.inference_mode()
def bank(model, data, rng, device, batch):
    model.eval()
    source = prior(data)
    waves = align_trials(data, source, rng)
    features = []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
        for lo in range(0, len(waves), batch):
            x = torch.from_numpy(waves[lo:lo + batch]).to(device)
            features.append(model.features(x).float().cpu())
    z = torch.cat(features)
    mean, inv_std = z.mean(0), z.std(0).clamp_min(0.15).reciprocal()
    z = ((z - mean) * inv_std).clamp(-8, 8).numpy()
    global_weight = model.task.weight[:3].detach().float().cpu()
    global_bias = model.task.bias[:3].detach().float().cpu() + global_weight @ mean
    global_weight = global_weight / inv_std
    raw = center(data.x)
    identity = fingerprint(raw, covariance(raw))
    gallery, w, b, enrolled = [], [], [], []
    for person in np.unique(data.person):
        idx = (data.person == person) & (data.visit == "1")
        if not idx.any():
            raise ValueError(f"{person}: enrolment session 1 is missing")
        gallery.append(identity[idx].mean(0))
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=300).fit(z[idx], data.y[idx])
        if tuple(clf.classes_) != (0, 1, 2):
            raise ValueError(f"{person}: incomplete enrolment labels")
        w.append(clf.coef_.astype(np.float32))
        b.append(clf.intercept_.astype(np.float32))
        enrolled.append(True)
    g = np.stack(gallery)
    scale = 1 / np.maximum(g.std(0), 0.08)
    distances = np.sqrt(np.mean(((g[None] - g[:, None]) * scale) ** 2, axis=-1))
    offdiag = distances[~np.eye(len(g), dtype=bool)]
    radius = float(np.quantile(offdiag, 0.10) * 0.8) if len(offdiag) else 0.0
    return dict(
        covariance=torch.from_numpy(source), gallery=torch.from_numpy(g), identity_scale=torch.from_numpy(scale),
        identity_radius=radius, personal_weight=torch.from_numpy(np.stack(w)),
        personal_bias=torch.from_numpy(np.stack(b)), enrolled=torch.tensor(enrolled),
        feature_mean=mean, feature_inv_std=inv_std, weight=global_weight, bias=global_bias,
    )
