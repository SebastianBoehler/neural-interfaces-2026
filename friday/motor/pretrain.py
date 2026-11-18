"""Pretrain spatial waveform features on PhysioNet motor/rest targets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fit import sha
from temporal import TemporalDecoder

CLASSES = ("feet", "hands", "left_hand", "rest", "right_hand")
SHARDS = ("windows_1-10.npz", "windows_11-20.npz", "windows_21-30.npz")
PEOPLE = [f"PhysionetMI/{i}" for i in range(1, 31)]
KINDS = ("motor_transfer", "motor_null")


def read_source(path, smoke):
    manifest = dict(line.split()[::-1] for line in Path("physionet.sha256").read_text().splitlines())
    if set(manifest) != set(SHARDS):
        raise ValueError("unexpected set of PhysioNet shards")
    with np.load("proteus_windows.npz", allow_pickle=False) as z:
        channels = z["ch_names"]
    data, labels, people, runs = [], [], [], []
    for filename in SHARDS[:1] if smoke else SHARDS:
        shard = path / filename
        if sha(shard) != manifest[filename]:
            raise ValueError(f"PhysioNet shard mismatch {filename}")
        with np.load(shard, allow_pickle=False) as z:
            if (tuple(z["classes"]) != CLASSES or not np.array_equal(z["ch_names"], channels)
                    or z["interpolated"].size or set(z["task"]) != {"PhysionetMI"}):
                raise ValueError(f"unexpected PhysioNet sensor or label schema {filename}")
            data.append(z["X"])
            labels.append(z["y"])
            people.append(z["subject"])
            runs.append(z["run"])
    x, y, names, run = (np.concatenate(values) for values in (data, labels, people, runs))
    expected = PEOPLE[:10] if smoke else PEOPLE
    if (x.shape != (len(expected) * 174, 41, 480) or x.dtype != np.float32
            or not np.isfinite(x).all() or set(names) != set(expected) or set(y) != set(range(5))):
        raise ValueError("unexpected PhysioNet cohort or tensor")
    return x, y, names, run, expected


def permute_labels(y, people, runs, seed):
    altered = y.copy()
    rng = np.random.default_rng(seed + 10000)
    for person in sorted(set(people)):
        for run in sorted(set(runs[people == person])):
            rows = (people == person) & (runs == run)
            altered[rows] = rng.permutation(y[rows])
            np.testing.assert_array_equal(np.sort(altered[rows]), np.sort(y[rows]))
    if not np.any(altered != y):
        raise ValueError("PhysioNet target permutation is inactive")
    return altered


def main(kind, seed, path, smoke):
    if "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("expected Galvani A100")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    x, raw_y, people, runs, expected = read_source(path, smoke)
    shuffled = permute_labels(raw_y, people, runs, seed)
    y = torch.from_numpy(raw_y if kind == "motor_transfer" else shuffled).long().cuda()
    model = TemporalDecoder("chronological").cuda()
    head = nn.Linear(32, len(CLASSES)).cuda()
    params = [model.spatial, *model.encoder.parameters(), *head.parameters()]
    initial = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.001)
    rng = np.random.default_rng(seed)
    bins = torch.cat([torch.arange(4, 180, device="cuda") / 4,
                      torch.arange(1, 45, device="cuda").repeat(4)])
    with torch.no_grad():
        spectrum = model.spectrum(torch.from_numpy(x).cuda()) * ((bins >= 7) & (bins < 30))
    steps = 4 if smoke else 1200
    for i in range(steps):
        rows = rng.choice(len(x), size=96, replace=False)
        loss = F.cross_entropy(head(model.encode(spectrum[rows])), y[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if i in (0, steps - 1):
            if any(p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0
                   for p in params):
                raise ValueError("inactive or nonfinite pretraining gradient")
        optimizer.step()
    if any(torch.equal(p, before) for p, before in zip(params, initial)):
        raise ValueError("unexpected unchanged pretraining tensor")
    if not loss.isfinite():
        raise ValueError("nonfinite source CE")
    stem = f"{'smoke_' if smoke else ''}{kind}_{seed}_encoder"
    checkpoint = Path(stem + ".pt")
    model.cpu()
    torch.save(dict(spatial=model.spatial.detach(), encoder=model.encoder.state_dict()), checkpoint)
    checkpoint.with_suffix(".json").write_text(json.dumps(
        dict(kind=kind, seed=seed, steps=steps, checkpoint=sha(checkpoint), actor=sha(Path("pretrain.sha256")),
             inputs=sha(Path("physionet.sha256")), people=expected), indent=2) + "\n")
    print(f"{kind} seed {seed}, {steps} updates, source CE {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("seed", type=int, choices=(460, 1460))
    parser.add_argument("source", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.kind, args.seed, args.source, args.smoke)
