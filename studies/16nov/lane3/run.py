"""§292C: person-disjoint CBraMod factorization; no held-person scoring."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import torch
from torch import nn
import deep
import protocol
from factor import FactorNet, person_batches
from prep import ORDERS, aligned_rows, partition, ranks

BLOB = "a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3"
ARMS = ("off", "matched", "split")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_orders(n):
    with np.load(ROOT / "runs/shuffle_perms.npz") as f:
        return {o: ranks(f[f"{o[:-1]}|{o[-1]}|all_loso"], n) for o in ORDERS}


def weight_path():
    from huggingface_hub import hf_hub_download
    repo, rev, _, _ = deep.PRETRAINED["cbramod"]
    path = hf_hub_download(repo, "model.safetensors", revision=rev, local_files_only=True)
    if sha(path) != BLOB:
        raise ValueError("pretrained backbone hash mismatch")
    return path


def validation(model, x, y, device):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(batch.to(device))[0].cpu() for batch in x.split(16)])
    if not torch.isfinite(logits).all() or set(y.tolist()) != {0, 1, 2}:
        raise ValueError("invalid validation logits or missing class")
    p = logits.argmax(1).numpy()
    return float(np.mean([(p[y == c] == c).mean() for c in range(3)]))


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(batch.to(device))[0].double().softmax(1).cpu() for batch in x.split(16)]).numpy()
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("invalid predictions")
    return p


def train(model, x, y, people, streams, val_x, val_y, seed, epochs, device):
    rng = np.random.default_rng(seed)
    rows = np.arange(len(y))
    batches = person_batches(rows, people, streams, np.random.default_rng(seed))
    head_params = list(model.head.parameters())
    if model.arm != "off":
        head_params += list(model.task.parameters()) + list(model.style.parameters())
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 1e-4},
                             {"params": head_params, "lr": 1e-3}], weight_decay=0.05)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-4, 1e-3], total_steps=epochs * len(batches))
    best, state, log, checked_aux = -1.0, None, [], False
    for epoch in range(epochs):
        model.train()
        current = person_batches(rows, people, streams, rng)
        if len(current) != len(batches):
            raise ValueError("sampler changed batch count")
        for b in current:
            ids = torch.as_tensor(people[b], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits, auxiliary = model(x[b].to(device), ids)
            loss = nn.functional.cross_entropy(logits, y[b].to(device)) + auxiliary
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            if model.arm == "split" and not checked_aux:
                grad = torch.autograd.grad(auxiliary, list(model.style.parameters()), retain_graph=True)
                if not all(torch.isfinite(g).all() for g in grad) or sum(float(g.abs().sum()) for g in grad) <= 0:
                    raise ValueError("auxiliary loss does not update style branch")
                checked_aux = True
            loss.backward()
            opt.step()
            schedule.step()
        value = validation(model, val_x, val_y, device)
        log.append(value)
        if value > best:
            best = value
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if state is None:
        raise ValueError("no validation checkpoint selected")
    model.load_state_dict(state)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("seed", type=int, choices=(0, 1))
    ap.add_argument("held", help="person name, e.g. P01")
    ap.add_argument("--smoke", action="store_true", help="one epoch; never evaluates held person")
    args = ap.parse_args()
    out = Path(os.environ["NI26_OUT"])
    if not out.is_dir():
        raise ValueError("NI26_OUT must be an existing isolated output directory")
    target = out / (f"smoke_{args.arm}_s{args.seed}_{args.held}.json" if args.smoke else
                    f"{args.arm}_s{args.seed}_{args.held}.npz")
    if target.exists():
        raise FileExistsError(target)
    t0 = time.time()
    checkpoint = weight_path()
    d = protocol.load()
    people = d["subject"]
    held = next((p for p in np.unique(people) if p.split("/")[-1] == args.held), None)
    train_rows, val_rows, test_rows = partition(people, held, args.seed)
    print("partition", len(train_rows), len(val_rows), len(test_rows), "val", people[val_rows[0]], flush=True)
    rank = get_orders(len(people))
    align = deep.AlignedNet(d, arch="cbramod", seed=args.seed, k=5.0)
    align.G = align.C[train_rows].mean(axis=0)
    fit_x = torch.from_numpy(aligned_rows(align, train_rows, rank["within0"]))
    val_x = torch.from_numpy(aligned_rows(align, val_rows, rank["within0"]))
    # Never read held-person labels in this runner, including after checkpoint selection.
    fit_y = torch.from_numpy(d["y"][train_rows]).long()
    val_y = d["y"][val_rows]
    names = sorted(set(people[train_rows]))
    ids = np.array([names.index(p) for p in people[train_rows]], dtype=np.int64)
    torch.manual_seed(args.seed)
    backbone = deep.build("cbramod", d["X"].shape[1], d["X"].shape[2], pretrained=True)
    device = deep.device()
    backbone.eval().to(device)
    with torch.no_grad():
        before = backbone(fit_x[:2].to(device)).detach().cpu()
    model = FactorNet(backbone, args.arm).to(device)
    if args.arm == "off":
        model.eval()
        with torch.no_grad():
            after = model(fit_x[:2].to(device))[0].cpu()
        if not torch.equal(before, after):
            raise ValueError(f"off arm changes backbone, max diff {(before-after).abs().max().item()}")
    log = train(model, fit_x, fit_y, ids, d["stream"][train_rows], val_x, val_y,
                args.seed, 1 if args.smoke else 25, device)
    meta = {"arm": args.arm, "seed": args.seed, "held": args.held, "val_person": str(people[val_rows[0]]),
            "best_epoch": int(np.argmax(log)), "elapsed_s": time.time() - t0,
            "checkpoint_sha256": sha(checkpoint), "orders": list(ORDERS), "trainable": sum(p.numel() for p in model.parameters())}
    if args.smoke:
        with target.open("x") as f:
            json.dump(meta, f, indent=2)
    else:
        result = {"index": test_rows, "val_log": np.asarray(log), "metadata": json.dumps(meta)}
        for order in ORDERS:
            x = torch.from_numpy(aligned_rows(align, test_rows, rank[order]))
            result[f"proba_{order}"] = predict(model, x, device)
        with target.open("xb") as f:
            np.savez(f, **result)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
