"""Fit a complete waveform actor against precomputed fit-excluded stock posteriors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fit import (DATA_SHA, ORDER_SHA, ORDERS, PEOPLE, check_behavior, read_inputs,
                 sha, source_weights)
from source_labels import read_training_labels
from temporal import TemporalDecoder
from waveform import center

OBJECTIVES = ("true", "wrongswap", "ordinary")
TEACHER_SOURCE = "249acc834ea4c0c087cec55be3d155a96fbd15580c0a0f0b277fa6224d9afa4d"


def corrupt(teacher, labels):
    classes = torch.arange(3, device=teacher.device)
    swap = torch.where(classes == labels[:, None], classes, 3 - labels[:, None] - classes)
    matched = teacher.gather(-1, swap)
    before = teacher.argmax(-1) == labels
    after = matched.argmax(-1) == labels
    return torch.where((before == after).unsqueeze(-1), matched, teacher)


def read_teacher(people, streams, perms):
    path = Path("teacher.npz")
    receipt = json.loads(Path("teacher.json").read_text())
    expected = dict(bank=sha(path), data=DATA_SHA, orders_sha=ORDER_SHA, source=TEACHER_SOURCE,
                    orders=list(ORDERS), windows=len(people))
    if receipt != expected:
        raise ValueError("historical teacher provenance mismatch")
    with np.load(path, allow_pickle=False) as bank:
        if not np.array_equal(bank["person"], people):
            raise ValueError("teacher row ownership mismatch")
        scores, ages = bank["score"], bank["age"]
    if scores.shape != (8, len(people), 3) or not np.isfinite(scores).all():
        raise ValueError("invalid teacher probability bank")
    episodes = []
    for i, order in enumerate(ORDERS):
        perm = perms[order]
        local = [perm[streams[perm] == s] for s in sorted(set(streams))]
        for rows in local:
            np.testing.assert_array_equal(ages[i, rows], np.arange(len(rows)))
            episodes.append((i, rows))
    return torch.from_numpy(scores).cuda(), episodes, receipt


def fit(objective, seed, steps):
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("expected Ferranti H100")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    data_path = Path("proteus_windows.npz")
    x, people, streams, interfaces, perms = read_inputs(data_path, Path("shuffle_perms.npz"))
    teacher, episodes, teacher_receipt = read_teacher(people, streams, perms)
    train_rows = np.arange(len(x))
    labels = read_training_labels(data_path, train_rows, len(x))
    weights = torch.from_numpy(source_weights(labels, train_rows, streams, interfaces)).cuda()
    y = torch.from_numpy(labels).cuda()
    eeg = torch.from_numpy(x).cuda()
    model = TemporalDecoder("chronological").cuda()
    with torch.no_grad():
        spectrum = model.spectrum(eeg)
        altered = torch.stack([corrupt(logits, y) for logits in teacher])
        torch.testing.assert_close(altered.sort(dim=-1).values, teacher.sort(dim=-1).values, atol=0, rtol=0)
        torch.testing.assert_close(altered[:, train_rows, y], teacher[:, train_rows, y], atol=0, rtol=0)
        if not torch.equal(altered.argmax(-1) == y, teacher.argmax(-1) == y):
            raise ValueError("wrong-class control changed stock correctness")
        if torch.count_nonzero(altered != teacher) == 0:
            raise ValueError("wrong-class control inactive")

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    for update in range(steps):
        oi, rows = episodes[rng.integers(len(episodes))]
        z = model.encode(spectrum[rows])
        static = model.static(z)
        logits = model.sequence(z, static, model.writes(z, static))
        target = y[rows]
        if objective != "ordinary":
            base = teacher if objective == "true" else altered
            logits = base[oi, rows] + .20 * center(logits).clamp(-3, 3)
        main = F.cross_entropy(logits, target, reduction="none")
        static_ce = F.cross_entropy(static, target, reduction="none")
        template_error = (model.templates[target] - z.detach()).square().mean(dim=-1)
        loss = ((main + .25 * static_ce + .05 * template_error) * weights[rows]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update in (0, steps - 1):
            for name, param in model.named_parameters():
                grad = param.grad
                if grad is None or not grad.isfinite().all() or grad.abs().max() == 0:
                    raise ValueError(f"invalid gradient for {name}")
        optimizer.step()

    if not loss.isfinite() or any(torch.equal(initial[n], p) for n, p in model.named_parameters()):
        raise ValueError("nonfinite or inactive waveform fit")
    check_behavior(model.eval(), eeg[episodes[0][1][:24]])
    stem = f"{'smoke_' if steps == 4 else ''}{objective}_{seed}"
    path = Path(stem + ".pt")
    if path.exists():
        raise FileExistsError(path)
    torch.save(dict(model=model.cpu().state_dict(), architecture="chronological",
                    objective=objective, seed=seed), path)
    receipt = dict(objective=objective, seed=seed, architecture="chronological", steps=steps,
                   checkpoint=sha(path), actor=sha(Path("actor.sha256")), data=DATA_SHA, orders=ORDER_SHA,
                   teacher=teacher_receipt["bank"], train_people=list(PEOPLE), external_labels_opened=False)
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{objective} {seed}: {steps} updates, teacher, matched corruption and causal checks OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objective", choices=OBJECTIVES)
    parser.add_argument("seed", type=int, choices=(460, 1460))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    fit(args.objective, args.seed, 4 if args.smoke else 600)
