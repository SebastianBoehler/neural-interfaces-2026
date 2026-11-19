"""Build Team A's official solver package from explicitly permitted source rows."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from data import CORE, EXT, external_data, read_public, sha, source_rows
from fit import fit
from gallery import bank
from package import BASE, HERE, bundle, initialize


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--people", required=True)
    p.add_argument("--first-session-only", default="")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=os.environ.get("CS_DATA", HERE.parents[1] / "data/proteus_windows.npz"))
    p.add_argument("--external-dir", type=Path, action="append", default=[])
    p.add_argument("--external", default=os.environ.get("TEAM_A_CORPORA", ",".join((*CORE, "scherer2015"))))
    p.add_argument("--arch", choices=tuple(BASE), default=os.environ.get("TEAM_A_ARCH", "reve"))
    p.add_argument("--assets", type=Path, default=os.environ.get("TEAM_A_ASSETS", HERE / "assets"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--recipe", choices=("A1", "A2"), default="A1")
    p.add_argument("--transfer", type=Path, default=os.environ.get("TEAM_A_TRANSFER"))
    p.add_argument("--external-fit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--steps", type=int, default=1152)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse()
    started = time.monotonic()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Team A training requires a Slurm CUDA job")
    if args.external_fit and args.transfer:
        raise ValueError("A shared external fit must initialize directly from the pretrained release")
    if args.steps < 1 or args.epochs < 1 or args.batch < 3:
        raise ValueError("Positive epochs/steps and batch >= 3 required")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    people = args.people.split(",")
    first = args.first_session_only.split(",") if args.first_session_only else []
    if set(first) - set(people):
        raise ValueError("--first-session-only must name a subset of --people")
    data = read_public(args.data)
    public = data.subset(source_rows(data, people, first))
    corpora = tuple(args.external.split(","))
    if len(set(corpora)) != len(corpora) or set(corpora) - set((*CORE, *EXT)):
        raise ValueError(f"Unknown or repeated corpora: {corpora}")
    runtime = bundle(args.arch, args.out, args.assets, args.recipe)
    model, pretrained = initialize(runtime, args.arch, public.names, args.assets, args.recipe, args.checkpoint)
    external, digests, transferred = {}, {}, None
    recipe = args.recipe
    if args.transfer:
        payload = torch.load(args.transfer, map_location="cpu", weights_only=True)
        if (payload["arch"], payload["names"], payload["corpora"], payload["recipe"], payload["seed"]) != (
                args.arch, public.names, corpora, recipe, args.seed):
            raise ValueError("Transferred checkpoint architecture, montage, corpus roster, recipe or seed differs")
        transferred, digests = payload["model"], payload["inputs"]
        if not args.smoke and payload["smoke"]:
            raise ValueError("Cannot train a scored system from smoke weights")
    else:
        for name in corpora:
            external[name], pin = external_data(name, [*args.external_dir, args.data.parent], public.names,
                                                rng, smoke=args.smoke)
            digests.update(pin)
            print(name, len(external[name].y), external[name].classes, flush=True)
    steps = (len(corpora) if args.smoke else args.steps) if not args.transfer else 0
    epochs = 0 if args.external_fit else 1 if args.smoke else args.epochs
    fit(model, external, public, rng, device, args.batch, steps, epochs, transferred, args.smoke, recipe)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    artifact = dict(arch=args.arch, names=public.names, model=state, corpora=corpora,
                    inputs=digests, recipe=recipe, seed=args.seed, smoke=args.smoke)
    if not args.external_fit:
        artifact["bank"] = bank(model, public, rng, device, args.batch, recipe)
    path = args.out / "weights.pt"
    torch.save(artifact, path)
    meta = dict(arch=args.arch, people=[] if args.external_fit else people, first_session_only=first,
                init_sha=pretrained, public_sha=None if args.external_fit else sha(args.data),
                transfer_sha=sha(args.transfer) if args.transfer else None, inputs=digests, corpora=corpora,
                steps=steps, epochs=epochs, seed=args.seed, recipe=recipe, smoke=args.smoke,
                weights_sha=sha(path), seconds=round(time.monotonic() - started, 2))
    (args.out / "training.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta["weights_sha"], f"fit completed in {meta['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
