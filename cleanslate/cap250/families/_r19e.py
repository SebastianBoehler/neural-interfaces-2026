"""Shared R19e ordinary-net recipe: fit one `scripts/sealed_r19e.json` member group on source rows and export it.

Mirrors `scripts/build_sealed.py:build` for the AlignedNet members only (no v2 core, no session EM, no rebalance),
fitted on the permitted source rows in their original order. The weights use the cap250 keys `net<I>_*`.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from components import FAMILIES, ORDINARY  # noqa: E402
from source import ASSETS, ROOT, family_args, load_source  # noqa: E402

CONFIG = ROOT / "scripts/sealed_r19e.json"
GROUPS = tuple(ORDINARY)  # family names in group order 0..7


def setup_paths():
    for p in ("2026-competition/tracks/bci_decoding", "scripts", "experiments"):
        sys.path.insert(0, str(ROOT / p))


def member(family, seed):
    cfg = json.loads(CONFIG.read_text())
    group = cfg["members"][GROUPS.index(family)]
    if seed not in group["seeds"]:
        raise ValueError(f"{family}: seed {seed} not in R19e group {group['seeds']}")
    kw = {**cfg["net_defaults"], **group.get("kw", {})}
    if "init" in kw:  # repo asset, resolved under CAP250_ASSETS; {seed} is filled by deep.AlignedNet
        kw["init"] = str(ASSETS / kw["init"])
        if not Path(kw["init"].format(seed=seed)).is_file():
            raise FileNotFoundError(kw["init"].format(seed=seed))
    return group["arch"], group["align"], kw


def export(m):
    from add_fbcnet import fbc_params
    from build_sealed import fbl_params, sinc_params
    from train_ensemble import net_params
    m.net.to("cpu")
    fn = {"fbcnet": fbc_params, "fblight": fbl_params, "sincshallow": sinc_params}.get(m.arch, net_params)
    return fn(m)


def main(family):
    args = family_args(f"R19e {family}", range(FAMILIES[family].seeds))
    setup_paths()
    import torch
    from deep import AlignedNet, device
    arch, align, kw = member(family, args.seed)
    if os.environ.get("CAP250_SMOKE") == "1":
        kw["epochs"] = 2
    elif device() != "cuda":
        raise RuntimeError("R19e fits require CUDA (set NI26_DEVICE=cuda inside Slurm)")
    d = load_source(args.people, args.first_session_only, args.data)
    rows = np.arange(len(d["y"]))
    net = AlignedNet(d, arch=arch, align=align, seed=args.seed, **kw)
    t0 = time.time()
    net.fit(rows)
    seconds = round(time.time() - t0, 1)
    index = ORDINARY[family] + args.seed
    params = export(net)
    if not {"align", "k", "G", "scale", "shrink"} <= set(params):
        raise ValueError(f"{family}: unexpected export keys {sorted(params)}")
    np.savez(args.out / "weights.npz", **{f"net{index}_{k}": v for k, v in params.items()})
    diag = args.out / "diagnostics"
    diag.mkdir()
    (diag / "fit.json").write_text(json.dumps(dict(
        family=family, seed=args.seed, net_index=index, arch=arch, align=align, kw=kw,
        n_rows=len(rows), fit_seconds=seconds, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), indent=1, default=str))
    print(f"PASS {family} s{args.seed} net{index} rows={len(rows)} fit={seconds}s", flush=True)
