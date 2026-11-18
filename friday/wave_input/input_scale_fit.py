"""Fit matched F11 waveform controls on source-only episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fit import DATA_SHA, ORDER_SHA, PEOPLE, read_inputs, sha, source_hash, source_weights
from input_scale import (KINDS, PSEUDOWINDOWS, check_behavior, fit_reference, power_hash,
                         scale_episode, window_power)
from raw import MaskedObjective, RawDecoder
from source_labels import read_training_labels


def model_hash(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(np.asarray(value.detach().cpu().numpy(), dtype="<f4").tobytes())
    return digest.hexdigest()


def source_sessions(streams, people, src):
    roster = sorted(set(streams[src]))
    sessions = [np.flatnonzero(streams[src] == stream) for stream in roster]
    owners = [str(people[src[rows[0]]]) for rows in sessions]
    candidates = [[i for i, owner in enumerate(owners) if owner != who] for who in owners]
    if any(not slots for slots in candidates) or any(len(rows) < 96 for rows in sessions):
        raise ValueError("matched sensor histories require distinct, sufficiently long source sessions")
    return roster, sessions, candidates


def fit(kind, seed, held, smoke):
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("F11 fit requires Ferranti H100")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    data_path = Path("proteus_windows.npz")
    order_path = Path("shuffle_perms.npz")
    x, people, streams, interfaces, _ = read_inputs(data_path, order_path)
    src = np.flatnonzero(people != held)
    participants = sorted(set(people[src]))
    if participants != sorted(set(PEOPLE) - {held}) or len(participants) != 9:
        raise ValueError("invalid F11 held-person exclusion")
    roster, sessions, candidates = source_sessions(streams, people, src)
    labels = np.full(len(x), -1, np.int64)
    labels[src] = read_training_labels(data_path, src, len(x))
    weights = torch.from_numpy(source_weights(labels, src, streams, interfaces)[src]).cuda()
    y = torch.from_numpy(labels[src]).cuda()
    source_eeg = torch.from_numpy(x[src]).cuda()
    power = window_power(source_eeg)
    reference = fit_reference(power, weights)
    donor_stream = str(roster[0])
    donor_person = donor_stream.split("|", 1)[0]
    if donor_person == held or donor_person not in participants:
        raise ValueError("saved donor entered the held fold")
    donor_power = power[sessions[0]]
    model = RawDecoder().cuda()
    initial_model = model_hash(model)
    auxiliary = MaskedObjective(model, "raw_supervised", seed, "cuda")
    optimizer = torch.optim.AdamW([*model.parameters(), *auxiliary.head.parameters()],
                                  lr=.001, weight_decay=.001)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    rng = np.random.default_rng(seed)
    donor_rng = np.random.default_rng(seed + 30000)
    schedule = hashlib.sha256()
    donor_schedule = hashlib.sha256()
    max_gain = torch.zeros((), device="cuda", dtype=torch.float64)
    steps = 4 if smoke else 600
    for update in range(steps):
        episode = rng.integers(len(sessions))
        rows = rng.choice(sessions[episode], size=96, replace=False)
        donor = donor_rng.choice(candidates[episode])
        alternate = donor_rng.choice(sessions[donor], size=96, replace=False)
        schedule.update(np.asarray(src[rows], dtype="<i8").tobytes())
        donor_schedule.update(np.asarray(src[alternate], dtype="<i8").tobytes())
        history_rows = rows if kind == KINDS[0] else alternate
        scaled, gain = scale_episode(source_eeg[rows], reference, power[history_rows])
        max_gain = torch.maximum(max_gain, (gain - 1).abs().max())
        batch = model.prepare(scaled)
        latent = model.encode(batch)
        static = model.static(latent)
        logits = model.sequence(latent, static, model.writes(latent, static))
        train_y = y[rows]
        ce = F.cross_entropy(logits, train_y, reduction="none")
        ce_static = F.cross_entropy(static, train_y, reduction="none")
        template_error = (model.templates[train_y] - latent.detach()).square().mean(dim=-1)
        task = ((ce + .25 * ce_static + .05 * template_error) * weights[rows]).mean()
        latent_loss = auxiliary(model, batch)
        if update == 0:
            gradient = torch.autograd.grad(latent_loss, model.stem.weight, allow_unused=True,
                                           retain_graph=True)[0]
            if gradient is not None:
                raise ValueError("masked control was required to detach from the raw encoder")
        optimizer.zero_grad(set_to_none=True)
        loss = task + .20 * latent_loss
        loss.backward()
        if update in (0, steps - 1):
            for name, param in model.named_parameters():
                grad = param.grad
                if grad is None or not grad.isfinite().all() or grad.abs().max() == 0:
                    raise ValueError(f"invalid F11 gradient {name}")
        optimizer.step()
        auxiliary.update(model)
    active_gain = float(max_gain)
    if (not torch.isfinite(loss) or not 1e-4 < active_gain <= 1
            or any(torch.equal(initial[name], p) for name, p in model.named_parameters())):
        raise ValueError("inactive or nonfinite F11 fit")
    model.eval()
    probe = sessions[0][:24]
    check_behavior(model, source_eeg[probe], reference, power[probe])
    # The other fold-local source session provides a different person's sensor history.
    check_behavior(model, source_eeg[probe], reference, power[sessions[candidates[0][0]][:24]])
    prefix = "smoke_" if smoke else ""
    path = Path(f"{prefix}{kind}_{seed}_{held}.pt")
    if path.exists():
        raise FileExistsError(path)
    torch.save(dict(model=model.cpu().state_dict(), kind=kind, held=held, donor_stream=donor_stream,
                    reference=reference.cpu(), donor_power=donor_power.cpu()), path)
    receipt = dict(kind=kind, seed=seed, held=held, steps=steps, device="cuda",
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   actor=sha(Path("actor.sha256")), checkpoint=sha(path), data=DATA_SHA, orders=ORDER_SHA,
                   held_labels_opened=False, train_people=[str(p) for p in participants],
                   train_rows=source_hash(src), reference_rows=source_hash(src),
                   donor_stream=donor_stream, donor_person=donor_person, donor_rows=source_hash(src[sessions[0]]),
                   reference_sha256=power_hash(reference), donor_power_sha256=power_hash(donor_power),
                   initial_model=initial_model, episode_rows=schedule.hexdigest(),
                   donor_episode_rows=donor_schedule.hexdigest(),
                   prefix_windows=PSEUDOWINDOWS, active_gain=active_gain)
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{kind} {seed} {held}: {steps} source updates, matched schedules and causal guards OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("seed", type=int, choices=(460, 1460))
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    fit(args.kind, args.seed, args.held, args.smoke)
