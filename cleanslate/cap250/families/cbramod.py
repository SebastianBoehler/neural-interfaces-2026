"""Fit a CBraMod-C slot member: external stage A reused, 25-epoch source stage B (laneC/train.py::stage_b), export_blob half."""

import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _aux_cb import checkpoint, make_public, stage_a_dir  # noqa: E402
from source import ROOT, family_args  # noqa: E402

sys.path[:0] = [str(ROOT / "submissions/final_stack"), str(ROOT / "laneC2")]
import train_cbramod  # noqa: E402  (Lane C2 patches onto Lane C; fails loudly if one is inactive)
from cb_export import export_blob  # noqa: E402

lane_c = train_cbramod.lane_c


def fit(args):
    smoke = os.environ.get("CAP250_SMOKE") == "1"
    if not torch.cuda.is_available():
        raise RuntimeError("cbramod fit requires Slurm CUDA")
    blob = checkpoint()
    if not blob.is_file():
        raise FileNotFoundError(f"set CAP250_CB_CHECKPOINT to the pinned CBraMod safetensors: {blob}")
    t0 = time.time()
    lane_c.public = make_public(train_cbramod.data, args.people, args.first_session_only, args.data)
    with TemporaryDirectory(prefix="cb-", dir=args.out.parent) as tmp:
        work = Path(tmp)
        suffix = "_canary" if smoke else ""
        for kind in (".pt", ".json"):
            src = stage_a_dir() / f"external_C_{args.seed}{kind}"
            if not src.is_file():
                raise FileNotFoundError(f"missing external stage A file {src}")
            os.symlink(src, work / f"external_C_{args.seed}{suffix}{kind}")
        ns = argparse.Namespace(root=ROOT, work=work, checkpoint=blob)
        lane_c.stage_b(ns, "C", args.seed, "all", quick=smoke, full=True)
        stem = f"reve_full_C_{args.seed}_all{'_canary' if smoke else ''}"
        ckpt = torch.load(work / f"{stem}.pt", map_location="cpu", weights_only=True)
        torch.save(export_blob(ckpt, half=True), args.out / f"cb_s{args.seed}.pt")
    diagnostics = args.out / "diagnostics"
    diagnostics.mkdir()
    rows = lane_c.public.state["rows"]
    (diagnostics / "fit.json").write_text(
        f'{{"seed": {args.seed}, "rows": {rows}, "smoke": {str(smoke).lower()}, "gpu": "{torch.cuda.get_device_name()}", "sec": {round(time.time() - t0)}}}\n')


if __name__ == "__main__":
    fit(family_args(__doc__, (0, 1, 2)))
