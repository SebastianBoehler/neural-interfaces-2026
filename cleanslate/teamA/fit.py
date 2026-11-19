"""Supervised external transfer and full-backbone target fine-tuning."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import TASK, align_trials, prior, virtual_subjects


def sample_weights(data):
    keys = [f"{p}/{v}/{i}/{y}" for p, v, i, y in zip(data.person, data.visit, data.interface, data.y)]
    _, codes, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return (1.0 / counts[codes]) / len(counts)


def augment(x, mixes, rng):
    if rng.random() < 0.8:
        # One coherent virtual subject applies to all members of this batch.
        transform = mixes[rng.integers(len(mixes))]
        x = x + rng.uniform(0.05, 0.24) * (transform @ x - x)
    x = x * torch.empty((len(x), 1, 1), device=x.device).uniform_(0.75, 1.3)
    keep = (torch.rand((len(x), x.shape[1], 1), device=x.device) > 0.03).to(x.dtype)
    offset = int(rng.integers(-22, 23))
    wave = F.pad(x * keep, (22, 22), mode="reflect")[:, :, 22 + offset:502 + offset]
    return wave + 0.025 * torch.randn_like(wave)


def loss_for_external(model, domain_head, features, y, task_lookup):
    logits = model.task(features)
    local = F.cross_entropy(domain_head(features), y)
    if task_lookup is None:
        return local
    return local + 0.75 * F.cross_entropy(logits, task_lookup[y])


def check_updates(parameters, start):
    for name, initial in start.items():
        result = (parameters[name].detach().float().flatten()[::257] - initial).abs().max().item()
        if not 0 < result < float("inf"):
            raise RuntimeError(f"End-to-end gradient did not update {name}")


def fit(model, external, public, rng, device, batch, steps, epochs, transferred=None, smoke=False):
    bf16 = torch.cuda.is_bf16_supported()
    model.to(device).train()
    named = dict(model.encoder.named_parameters())
    keys = list(named)
    start = {key: named[key].detach().float().flatten()[::257].clone() for key in (keys[0], keys[-1])}
    for param in model.encoder.parameters():
        if not param.requires_grad:
            raise RuntimeError("Every backbone parameter must be trainable")

    if transferred is not None:
        model.load_state_dict(transferred, strict=True)
        start = {key: named[key].detach().float().flatten()[::257].clone() for key in start}
    if steps:
        ext_mixes = {k: torch.from_numpy(virtual_subjects(v, limit=12)).to(device)
                     for k, v in external.items()}
        aligned = {name: align_trials(data, prior(data), rng) for name, data in external.items()}
        probs = {name: sample_weights(data) for name, data in external.items()}
        private = nn.ModuleDict({k: nn.Linear(160, len(v.classes)) for k, v in external.items()}).to(device)
        task_lookup = {}
        for name, data in external.items():
            mapped = [TASK.get(c) for c in data.classes]
            task_lookup[name] = (torch.tensor(mapped, device=device)
                                 if None not in mapped and len(set(mapped)) > 1 else None)
        opt = torch.optim.AdamW(
            [{"params": model.encoder.parameters(), "lr": 5e-5},
             {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")] + list(private.parameters()),
              "lr": 8e-4}],
            weight_decay=0.03, fused=True,
        )
        corpora = list(external)
        losses = []
        for step in range(steps):
            domain = corpora[step % len(corpora)]
            d = external[domain]
            rows = rng.choice(len(d.y), batch, p=probs[domain])
            x = torch.from_numpy(aligned[domain][rows]).to(device)
            y = torch.as_tensor(d.y[rows], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                f = model.features(augment(x, ext_mixes[domain], rng))
                loss = loss_for_external(model, private[domain], f, y, task_lookup[domain])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            opt.step()
            losses.append(loss.detach())
            if (step + 1) % 128 == 0 or step + 1 == steps:
                print(f"external step {step + 1}/{steps} loss {torch.stack(losses).mean().item():.4f}", flush=True)
                losses.clear()
        del aligned, probs, private, opt
        check_updates(named, start)

    if not epochs:
        return
    mixes = torch.from_numpy(virtual_subjects(public)).to(device)
    source = prior(public)
    weights = sample_weights(public)
    opt = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": 3e-5},
         {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 7e-4}],
        weight_decay=0.04, fused=True,
    )
    steps_per_epoch = 2 if smoke else max(1, len(public.y) // batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch, eta_min=8e-6)
    for epoch in range(epochs):
        x = align_trials(public, source, rng)
        losses = []
        for _ in range(steps_per_epoch):
            rows = rng.choice(len(public.y), batch, p=weights)
            wave = torch.from_numpy(x[rows]).to(device)
            label = torch.as_tensor(public.y[rows], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = F.cross_entropy(model.task(model.features(augment(wave, mixes, rng)))[:, :3],
                                       label, label_smoothing=0.035)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            losses.append(loss.detach())
        print(f"public epoch {epoch + 1}/{epochs} loss {torch.stack(losses).mean().item():.4f}", flush=True)
    check_updates(named, start)
