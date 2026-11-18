"""Fit source-excluded waveform actors, and replay saved actors on pinned orders."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from source_labels import read_training_labels
from waveform import Decoder
from temporal import KINDS as TEMPORAL_KINDS, SourceJitter, TemporalDecoder


DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
PEOPLE = tuple(f"P{i:02}" for i in range(1, 11))
MOTOR_KINDS = ("motor_transfer", "motor_null")
KINDS = ("prototype", "centered", *TEMPORAL_KINDS, *MOTOR_KINDS)


def make_decoder(kind):
    if kind in MOTOR_KINDS:
        kind = "chronological"
    return TemporalDecoder(kind) if kind in TEMPORAL_KINDS else Decoder(kind)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def read_inputs(data_path, order_path):
    if sha(data_path) != DATA_SHA or sha(order_path) != ORDER_SHA:
        raise ValueError("waveform data or order hash mismatch")
    with np.load(data_path, allow_pickle=False) as data:
        x = data["X"]
        people = np.array([p.split("/")[-1] for p in data["subject"]])
        streams = np.char.add(np.char.add(people, "|"), data["session"])
        interfaces = data["interface"]
    if (x.shape != (2655, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all()
            or tuple(np.unique(people)) != PEOPLE or len(set(streams)) != 14):
        raise ValueError("unexpected official EEG or metadata schema")
    with np.load(order_path, allow_pickle=False) as orders:
        perms = {k: orders[f"{k[:-1]}|{k[-1]}|all_loso"] for k in ORDERS}
    if any(not np.array_equal(np.sort(perm), np.arange(len(x))) for perm in perms.values()):
        raise ValueError("invalid order permutation")
    return x, people, streams, interfaces, perms


def initialize_motor(model, kind, seed):
    """Apply only the authenticated encoder parameters, never its source task head."""
    root = Path("pretrain")
    checkpoint = root / f"{kind}_{seed}_encoder.pt"
    info = json.loads(checkpoint.with_suffix(".json").read_text())
    expected = dict(kind=kind, seed=seed, steps=1200, checkpoint=sha(checkpoint),
                    actor=sha(root / "pretrain.sha256"), inputs=sha(root / "physionet.sha256"),
                    people=[f"PhysionetMI/{i}" for i in range(1, 31)])
    if info != expected:
        raise ValueError("pretrained encoder provenance mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"spatial", "encoder"}:
        raise ValueError("unexpected pretrained encoder tensors")
    with torch.no_grad():
        model.spatial.copy_(payload["spatial"])
    model.encoder.load_state_dict(payload["encoder"])
    return dict(pretrain=expected["checkpoint"], pretrain_source=expected["actor"])


def source_weights(labels, src, streams, interfaces):
    weights = np.zeros(len(labels), np.float32)
    for s in np.unique(streams[src]):
        rows = src[streams[src] == s]
        for interface in np.unique(interfaces[rows]):
            for c in range(3):
                cell = rows[(interfaces[rows] == interface) & (labels[rows] == c)]
                if not len(cell):
                    raise ValueError(f"missing class {c} in {s}/{interface}")
                weights[cell] = len(rows) / (6 * len(cell))
    return weights


@torch.no_grad()
def check_behavior(model, x):
    encoded = model.encode(model.spectrum(x))
    static = model.static(encoded)
    writes = model.writes(encoded, static)
    logits = model.sequence(encoded, static, writes)
    state, steps = None, []
    for window in x.split(1):
        result, state = model.step(window, state)
        steps.append(result)
    restart, _ = model.step(x[:1])
    torch.testing.assert_close(torch.cat(steps), logits, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(model.sequence(encoded[:5], static[:5], writes[:5]),
                               logits[:5], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(steps[0], restart, atol=0, rtol=0)
    torch.testing.assert_close(logits[0], static[0], atol=0, rtol=0)
    if (logits - static).abs().max().item() <= 1e-4 or not logits.isfinite().all().item():
        raise ValueError("session history is inactive or nonfinite")


def fit(kind, seed, held, smoke, data_path, order_path, device):
    if device == "cuda" and "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("expected Ferranti H100")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    x, people, streams, interfaces, _ = read_inputs(data_path, order_path)
    src = np.flatnonzero(people != held)
    train_people = np.unique(people[src]).tolist()
    if held not in PEOPLE or len(train_people) != 9 or held in train_people:
        raise ValueError("invalid held-person exclusion")
    labels = np.full(len(x), -1, np.int64)
    labels[src] = read_training_labels(data_path, src, len(x))
    weights = torch.from_numpy(source_weights(labels, src, streams, interfaces)[src]).to(device)
    y = torch.from_numpy(labels[src]).to(device)
    sessions = [np.flatnonzero(streams[src] == s) for s in np.unique(streams[src])]
    model = make_decoder(kind).to(device)
    provenance = initialize_motor(model, kind, seed) if kind in MOTOR_KINDS else {}
    source_eeg = torch.from_numpy(x[src]).to(device)
    steps = 4 if smoke else 600
    jitter = SourceJitter(kind, seed, steps) if kind in TEMPORAL_KINDS[2:] else None
    with torch.no_grad():
        spectrum = model.spectrum(source_eeg) if jitter is None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    rng = np.random.default_rng(seed)
    for update in range(steps):
        rows = rng.choice(sessions[rng.integers(len(sessions))], size=96, replace=False)
        batch = (spectrum[rows] if jitter is None else
                 model.spectrum(jitter(source_eeg[rows], update)))
        z = model.encode(batch)
        static = model.static(z)
        logits = model.sequence(z, static, model.writes(z, static))
        train_y = y[rows]
        ce = F.cross_entropy(logits, train_y, reduction="none")
        ce_static = F.cross_entropy(static, train_y, reduction="none")
        template_error = (model.templates[train_y] - z.detach()).square().mean(dim=-1)
        loss = ((ce + .25 * ce_static + .05 * template_error) * weights[rows]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update in (0, steps - 1):
            for name, param in model.named_parameters():
                grad = param.grad
                if grad is None or not grad.isfinite().all() or grad.abs().max() == 0:
                    raise ValueError(f"invalid gradient for {name}")
        optimizer.step()
    if not loss.isfinite() or any(torch.equal(initial[n], p) for n, p in model.named_parameters()):
        raise ValueError("inactive or nonfinite fit")
    model.eval()
    check_behavior(model, source_eeg[sessions[0][:24]])
    donor_stream = str(sorted(set(streams[src]))[0])
    prefix = "smoke_" if smoke else ""
    path = Path(f"{prefix}{kind}_{seed}_{held}.pt")
    if path.exists():
        raise FileExistsError(path)
    torch.save(dict(model=model.cpu().state_dict(), donor_stream=donor_stream,
                    kind=kind, held=held), path)
    receipt = dict(kind=kind, seed=seed, held=held, steps=steps, device=str(device),
                   gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                   torch=torch.__version__, actor=sha(Path("actor.sha256")), checkpoint=sha(path),
                   data=DATA_SHA, orders=ORDER_SHA, held_labels_opened=False,
                   train_people=train_people, train_rows=source_hash(src), **provenance)
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{kind} {seed} {held}: {steps} source updates and causal guards OK", flush=True)


@torch.no_grad()
def score_checkpoint(path, x, people, streams, perms, device="cpu"):
    payload = torch.load(path, weights_only=True, map_location=device)
    held, donor = payload["held"], payload["donor_stream"]
    if held not in PEOPLE or donor.split("|")[0] == held or donor not in streams:
        raise ValueError("invalid checkpoint fold or donor")
    model = make_decoder(payload["kind"]).to(device).eval()
    model.load_state_dict(payload["model"])
    idx = np.flatnonzero(people == held)
    donor_idx = np.flatnonzero(streams == donor)
    donor_z = model.encode(model.spectrum(torch.from_numpy(x[donor_idx]).to(device)))
    donor_writes = model.writes(donor_z, model.static(donor_z))
    out = {arm: np.full((8, len(idx), 3), np.nan, np.float32)
           for arm in ("genuine", "donor", "blank")}
    age = np.full((8, len(idx)), -1, np.int64)
    check_behavior(model, torch.from_numpy(x[idx[:20]]).to(device))
    for i, order in enumerate(ORDERS):
        rows = perms[order]
        for stream in sorted(set(streams[idx])):
            stream_rows = rows[streams[rows] == stream]
            pos = np.searchsorted(idx, stream_rows)
            z = model.encode(model.spectrum(torch.from_numpy(x[stream_rows]).to(device)))
            static = model.static(z)
            own = model.writes(z, static)
            other = donor_writes[torch.arange(len(z), device=device) % len(donor_writes)]
            out["blank"][i, pos] = static.cpu().numpy()
            out["genuine"][i, pos] = model.sequence(z, static, own).cpu().numpy()
            out["donor"][i, pos] = model.sequence(z, static, other).cpu().numpy()
            age[i, pos] = np.arange(len(pos))
    if any(not np.isfinite(v).all() for v in out.values()) or (age < 0).any():
        raise ValueError("incomplete held bank")
    for logits in out.values():
        np.testing.assert_array_equal(logits[age == 0], out["blank"][age == 0])
    return dict(index=idx, age=age, **out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("seed", type=int, choices=(460, 1460))
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    fit(args.kind, args.seed, args.held, args.smoke, Path("proteus_windows.npz"),
        Path("shuffle_perms.npz"), "cuda")
