"""Train the fixed REVE D1 smoke and emit label-blind, row-aligned predictions."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from data import EXTERNAL, FULL_ARMS, FULL_ROLES, FULL_SEEDS, PEOPLE, external_source, public
from model import Member, augment, masked_ce
from oracle import require, sha

BATCH = 16
PARITY_ATOL = 2e-5  # batched vs singleton posterior agreement


def configure(seed):
    torch.manual_seed(20261119 + seed)
    require(torch.cuda.is_available(), "D1 requires a CUDA worker")
    torch.set_num_threads(2)
    return np.random.default_rng(20261119 + seed), torch.device("cuda")


def split_optimizer(model):
    body = list(model.body.parameters())
    head = list(model.body.final_layer.parameters()) + [model.body.cls_query_token] + list(model.motor.parameters())
    head_set = {id(p) for p in head}
    return torch.optim.AdamW([{"params": [p for p in body if id(p) not in head_set], "lr": 1e-4},
                              {"params": head, "lr": 1e-3}], weight_decay=0.05)


def step(model, opt, sched, x, pos, y, valid, rng, subtype=None, valid_motor=None):
    opt.zero_grad(set_to_none=True)
    x, pos = augment(x, pos, rng)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        task_logits, motor_logits = model(x, pos)
        loss = masked_ce(task_logits, y, valid) if len(valid) > 1 else task_logits.sum() * 0
        if valid_motor:
            known = subtype >= 0
            if known.any():
                loss = loss + masked_ce(motor_logits[known], subtype[known], valid_motor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
    opt.step()
    sched.step()
    return float(loss.detach())


def stage_a(args, arm, seed, quick):
    rng, device = configure(seed)
    model = Member(args.checkpoint).to(device).train()
    sources = [k for k in EXTERNAL if k != "scherer2015" or arm == "C"]
    domains = []
    info = {}
    for corpus in sources:
        x, y, motor, names, people, pins = external_source(args.root, corpus, quick)
        domains.append((torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
                        torch.from_numpy(motor).to(device), model.positions(names, device),
                        np.unique(y).tolist(), np.unique(motor[motor >= 0]).tolist()))
        info[corpus] = dict(rows=len(x), people=people, channels=names, inputs=pins)
    steps = len(domains) if quick else 960
    opt = split_optimizer(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for i in range(steps):
        x, y, sub, pos, valid, motors = domains[i % len(domains)]
        ix = rng.choice(len(y), BATCH, replace=False)
        loss = step(model, opt, sched, x[ix], pos, y[ix], valid, rng, sub[ix], motors)
        if (i + 1) % 120 == 0 or quick:
            print("stage A", arm, seed, i + 1, "loss", round(loss, 5), flush=True)
    path = args.work / f"external_{arm}_{seed}{'_canary' if quick else ''}.pt"
    torch.save(model.state_dict(), path)
    save_json(path.with_suffix(".json"), dict(arm=arm, seed=seed, steps=steps, sources=info,
              state_sha=sha(path), job=os.environ.get("SLURM_JOB_ID")))


def probabilities(model, x, pos):
    with torch.inference_mode():
        return np.concatenate([model(chunk, pos)[0][:, :3].softmax(1).cpu().numpy() for chunk in x.split(32)])


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def stage_b(args, arm, seed, held, quick=False, full=False):
    rng, device = configure(seed)
    model = Member(args.checkpoint).to(device)
    external = None
    if arm != "A":
        external = args.work / f"external_{arm}_{seed}{'_canary' if quick else ''}.pt"
        receipt = json.loads(external.with_suffix(".json").read_text())
        require(receipt["state_sha"] == sha(external), "stage A checkpoint changed")
        model.load_state_dict(torch.load(external, map_location="cpu", weights_only=True), strict=True)
    x, labels, train, query, orders, names, prior = public(args.root, held, full)
    pos = model.positions(names, device)
    x = [torch.from_numpy(view).to(device) for view in x]
    y = torch.from_numpy(labels).to(device)
    n = 1 if quick else 25
    opt = split_optimizer(model)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-4, 1e-3], epochs=n,
                                                steps_per_epoch=1 if quick else (len(y)+BATCH-1)//BATCH)
    model.train()
    for epoch in range(n):
        batches = rng.permutation(len(y))[:BATCH] if quick else rng.permutation(len(y))
        for j in range(0, len(batches), BATCH):
            ix = batches[j:j+BATCH]
            loss = step(model, opt, sched, x[epoch % 2][ix], pos, y[ix], [0, 1, 2], rng)
        if (epoch + 1) % 5 == 0 or quick:
            print("stage B", arm, seed, held, epoch + 1, "loss", round(loss, 5), flush=True)

    model.eval()
    out = dict(index=query, train=train)
    for name, (rows, view) in orders.items():
        tens = torch.from_numpy(view).to(device)
        p = probabilities(model, tens, pos)
        np.testing.assert_allclose(p[:3], probabilities(model, tens[:3], pos), atol=PARITY_ATOL, rtol=0)
        single = np.concatenate([probabilities(model, t[None], pos) for t in tens[:3]])
        np.testing.assert_allclose(p[:3], single, atol=PARITY_ATOL, rtol=0)
        require(np.isfinite(p).all(), "nonfinite posterior")
        out[name], out[f"{name}_delivery"] = p, rows
    stem = f"reve_{'full_' if full else ''}{arm}_{seed}_{held}{'_canary' if quick else ''}"
    bank = args.work / (stem + ".npz")
    state = args.work / (stem + ".pt")
    torch.save(dict(model=model.state_dict(), prior=torch.from_numpy(prior), channels=names), state)
    np.savez(bank, **out)
    result = dict(arm=arm, seed=seed, held=held, epochs=n, train_rows=len(train), query_rows=len(query),
                  bank_sha=sha(bank), state_sha=sha(state), checked_prefix=bool(orders),
                  external_state=None if external is None else str(external),
                  gpu=torch.cuda.get_device_name(), job=os.environ.get("SLURM_JOB_ID"))
    save_json(bank.with_suffix(".json"), result)
    print("PASS", result, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("canary", "canary-b", "external", "external-full", "proteus", "full"))
    p.add_argument("task", type=int)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    start = time.perf_counter()
    if args.mode in ("canary", "canary-b"):
        arm = "B" if args.mode == "canary-b" else "C"
        stage_a(args, arm, 0, True)
        stage_b(args, arm, 0, "P02", True)
    elif args.mode == "external":
        arm, seed = "BC"[args.task // 2], args.task % 2
        stage_a(args, arm, seed, False)
    elif args.mode == "external-full":
        arm, seed = divmod(args.task, FULL_SEEDS)
        stage_a(args, FULL_ARMS[arm], seed, False)
    elif args.mode == "proteus":
        if args.task == 24:
            stage_b(args, "B", 0, "returning")
        else:
            arm, i = divmod(args.task, 8)
            seed, person = divmod(i, 4)
            stage_b(args, "ABC"[arm], seed, PEOPLE[person])
    else:
        task, person = divmod(args.task, len(FULL_ROLES))
        arm, seed = divmod(task, FULL_SEEDS)
        stage_b(args, FULL_ARMS[arm], seed, FULL_ROLES[person], full=True)
    print("elapsed_s", round(time.perf_counter() - start, 2), flush=True)


if __name__ == "__main__":
    main()
