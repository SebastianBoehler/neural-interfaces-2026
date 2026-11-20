"""Train one nested LOPO attention model and stream every paired arm."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from base import D, fit_path, load_base, roster
from context import capture
from episodes import DEVICE, aligned, target_bank
from external import CORPORA, read_episodes
from fit import SEED, train
from package import ARMS, write_packages
from stream import load_solver, stream_sessions


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def sessions(d, person):
    return sorted(set(d["stream"][d["person"] == person]))


@torch.no_grad()
def audit(models, x, plan):
    _, streaming = D.official()
    rows = next(iter(plan.values()))[:18]
    probe = x[rows].to(DEVICE)
    ctx, blank = (streaming.fresh_model(models[a]) for a in ("context", "blank"))
    base = streaming.fresh_model(models["a0"])
    aligned_probe = aligned(probe)
    _, _, batch_logits = capture(base.nets, lambda: [net(aligned_probe) for net in base.nets])
    expected = batch_logits.softmax(-1).mean(1).log()
    seen = []
    for i in range(len(probe)):
        window = probe[i:i + 1]
        ctx.predict(window)
        blank.predict(window)
        seen.append(ctx.last.clone())
        torch.testing.assert_close(blank.last, expected[i], rtol=1e-4, atol=1e-4)
        if i == 0:
            torch.testing.assert_close(ctx.last, blank.last, rtol=0, atol=0)
    if torch.equal(ctx.last, blank.last) or ctx.count != len(probe):
        raise RuntimeError("the in-session context path was inactive")
    fresh = streaming.fresh_model(ctx)
    ctx.reset_state()
    for i in range(8):
        for m in (ctx, fresh):
            m.predict(probe[i:i + 1])
            if m.count != i + 1:
                raise RuntimeError("context state crossed a session boundary")
            torch.testing.assert_close(m.last, seen[i], rtol=0, atol=0)
    print("PASS zero-context logits, batched/online EA, reset, prefix and active context", flush=True)


def stream_all(root, person, d, smoke):
    D.official()
    x = torch.from_numpy(d["X"])
    packages = root / "packages" / person
    models = {a: load_solver(packages / a, d["ch_names"], DEVICE) for a in ARMS}
    plans = {order: D.plans(d, order, sessions(d, person)) for order in D.DEFAULT_ORDERS}
    first = next(iter(plans.values()))
    audit(models, x, first)
    warm = {s: rows[:18] for s, rows in list(first.items())[:1]}
    for model in models.values():
        stream_sessions(model, x, warm, DEVICE)
    out = {arm: {} for arm in ARMS}
    for order, plan in plans.items():
        if smoke:
            plan = warm
        shift = (int(person[1:]) + order) % len(ARMS)
        arms = ARMS[shift:] + ARMS[:shift]
        for arm in arms:
            for stream, (rows, labels, secs) in stream_sessions(models[arm], x, plan, DEVICE).items():
                session = stream.split("|")[1]
                for key, value in (("rows", rows), ("labels", labels), ("secs", secs)):
                    out[arm][f"{key}_{order}_{session}"] = value
                print(person, arm, order, session, len(rows), round(1000 * secs / len(rows), 3), "ms/window", flush=True)
        for stream in plan:
            key = f"labels_{order}_{stream.split('|')[1]}"
            np.testing.assert_array_equal(out["blank"][key], out["a0"][key])
            if out["context"][key][0] != out["blank"][key][0]:
                raise RuntimeError(f"first-window context parity failed in {stream}")
        if smoke:
            break
    for arm, output in out.items():
        directory = root / arm
        directory.mkdir(exist_ok=True)
        temporary = directory / f"{person}.tmp.npz"
        np.savez(temporary, **output)
        temporary.replace(directory / f"pred_new_{person}.npz")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("task", type=int)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M1 requires a Slurm GPU")
    torch.backends.cudnn.benchmark = False
    person = D.PEOPLE[args.task]
    epochs, orders, steps = (2, 1, 4) if args.smoke else (60, 8, 1200)
    base = load_base(args.root, (person,), epochs, DEVICE, bn="source", steps=0)
    d = D.load()
    inner = roster((person,))[:1] if args.smoke else roster((person,))
    target, extra, manifests = {}, {}, {}
    for episode_person in inner:
        rows = np.flatnonzero(d["person"] == episode_person)
        encoder = load_base(args.root, (person, episode_person), epochs, DEVICE, bn="source", steps=0)
        rng = np.random.default_rng([SEED, int(person[1:]), int(episode_person[1:])])
        target[episode_person] = target_bank(encoder, d["X"][rows], d["y"][rows], d["stream"][rows], rng, orders)
    del encoder
    for name in ("ds007221g",) if args.smoke else CORPORA:
        extra[name], manifests[name] = read_episodes(base, args.root / "external", name, args.smoke)
        print("external encoded", name, len(extra[name]), flush=True)
    head, visited = train(target, extra, {k: len(v["classes"]) for k, v in manifests.items()}, len(base.nets), steps)
    package_dir = args.root / "packages"
    fold = package_dir / person
    fold.mkdir(parents=True, exist_ok=True)
    context = fold / "context.pt"
    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, context)
    weights = fit_path(args.root, (person,)) / "weights.pt"
    write_packages(fold, weights, context)
    del base, extra, target
    receipt = dict(held=person, outer=roster((person,)), inner={e: roster((person, e)) for e in inner},
                   seed=SEED, epochs=epochs, steps=steps, draws=visited, external=manifests,
                   weights_sha256=sha(weights), context_sha256=sha(context),
                   solver_sha256=sha(fold / "context" / "submission.py"))
    (fold / "fit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    stream_all(args.root, person, d, args.smoke)


if __name__ == "__main__":
    main()
