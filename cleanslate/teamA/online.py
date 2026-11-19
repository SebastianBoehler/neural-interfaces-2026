"""Session-local prototype-free adaptation and open-set returning-person routing."""

import numpy as np
import torch
from torch.nn import functional as F


def normalize_features(x, mean, inv_std):
    return ((x - mean) * inv_std).clamp(-8, 8)


def balance(prob):
    """Soft uniform-prior assignments for the delivered prefix only."""
    q = prob.clamp_min(1e-5)
    for _ in range(6):
        q = q / q.sum(0, keepdim=True).clamp_min(1e-5)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-5)
    return q


class PrefixAdapter:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.features = []
        self.mean = torch.zeros_like(weight[0])
        self.scale = torch.ones_like(self.mean)
        self.offset = torch.zeros_like(self.mean)
        self.delta = torch.zeros_like(weight)
        self.prior = torch.zeros_like(bias)

    def read(self, feature):
        z = (feature - self.mean) * self.scale + self.offset
        return F.linear(z, self.weight + self.delta, self.bias) + self.prior

    def update(self, feature):
        self.features.append(feature)
        n = len(self.features)
        if n < 24 or n % 12:
            return
        x = torch.cat(self.features)
        share = n / (n + 96)
        self.mean = share * x.mean(0)
        self.scale = (1 - share + share * x.var(0, unbiased=False)).clamp(0.4, 3).rsqrt()
        z = (x - self.mean) * self.scale
        source = F.linear(x, self.weight, self.bias)
        p0 = source.softmax(-1)
        self.prior = (-0.5 * p0.mean(0).clamp_min(0.08).log()).clamp(0, 1)
        target = balance(p0.pow(1.5))
        confidence = (p0.amax(-1, keepdim=True) - 1 / 3).clamp_min(0)
        for _ in range(4):
            adjusted = z + self.offset
            w = self.weight + self.delta
            logp = (F.linear(adjusted, w, self.bias) + self.prior).log_softmax(-1)
            p = logp.exp()
            # Exact logit gradients of pseudo-label cross entropy and prediction entropy.
            entropy_grad = p * ((p * logp).sum(-1, keepdim=True) - logp)
            error = ((p - target) * confidence + 0.10 * entropy_grad) / n
            self.delta = (0.85 * self.delta - 0.06 * error.T @ adjusted).clamp(-0.12, 0.12)
            self.offset = (0.85 * self.offset - 0.12 * (error @ w).sum(0)).clamp(-0.2, 0.2)


class SessionDecoder:
    """Only immutable fit artifacts are shared by freshly copied sessions."""

    def __init__(self, net, bank, device):
        self.net, self.bank, self.device = net, bank, torch.device(device)
        self.reset_state()

    def __deepcopy__(self, memo):
        # The eval-mode encoder and fitted tensors are read-only. Each copy owns all writes.
        fresh = SessionDecoder(self.net, self.bank, self.device)
        memo[id(self)] = fresh
        return fresh

    def reset_state(self):
        self.age = 0
        self.covariance_sum = np.zeros_like(self.bank["covariance"])
        self.alignment = None
        self.identity_sum = np.zeros(self.bank["gallery"].shape[-1], np.float64)
        self.person = None
        self.streak = None
        self.adapter = PrefixAdapter(self.bank["weight"], self.bank["bias"])

    def _identify(self, raw, cov):
        gallery = self.bank["gallery"]
        if not len(gallery) or self.person is not None:
            return
        self.identity_sum += fingerprint(raw, cov)
        if self.age < 12 or self.age % 4:
            return
        difference = (gallery - self.identity_sum / self.age) * self.bank["identity_scale"]
        dist = (difference ** 2).mean(-1) ** 0.5
        ranked = np.argsort(dist)
        best = int(ranked[0])
        # Include the calibration gallery when looking for a competing identity.
        gap = dist[best] / max(dist[ranked[1]], 1e-6) if len(ranked) > 1 else float("inf")
        accepted = dist[best] < self.bank["identity_radius"] and gap < 0.80
        if accepted and self.streak == best and self.bank["enrolled"][best]:
            self.person = best
        self.streak = best if accepted else None

    @torch.inference_mode()
    def predict(self, x):
        if tuple(x.shape) != (1, 41, 480):
            raise ValueError(f"Expected (1, 41, 480), received {tuple(x.shape)}")
        raw = center(torch.as_tensor(x).float().cpu().numpy()[0])
        cov = covariance(raw)
        self.age += 1
        self.covariance_sum += cov
        if self.age % 8 == 1:
            self.alignment = update_transform(self.covariance_sum, self.age, self.bank["covariance"])
        self._identify(raw, cov)
        wave = torch.from_numpy((self.alignment @ raw)[None]).to(self.device)
        bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
            f = self.net.features(wave).float()
        f = normalize_features(f, self.bank["feature_mean"], self.bank["feature_inv_std"])
        self.adapter.update(f)
        logits = self.adapter.read(f)
        if self.person is not None:
            personal = F.linear(f, self.bank["personal_weight"][self.person], self.bank["personal_bias"][self.person])
            logits = 0.30 * logits + 0.70 * personal
        return logits.argmax(-1)
