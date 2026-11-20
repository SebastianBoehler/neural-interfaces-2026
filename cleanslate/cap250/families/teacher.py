"""Refit the cap250 external teacher: 12 EEGNeX + 12 IFNet m7r trunks, fold-fitted EA and logistic map.

The 24 trunks are external (seven non-PROTEUS MI corpora, no UDA). They are verified by SHA-256
against scripts/sealed_nxif_inputs.sha256 and by their training metadata before use. Only the
source-row causal EA reference (G, scale) and the balanced 38->3 logistic map (C .01) see PROTEUS.
Writes weights.npz (net34_{align,k,G,scale,shrink,jit_file}), teacher.pt (TorchScript with the map).
"""

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ASSETS, ROOT, family_args, load_source
from _teacher_lib import Teacher, Heads, ea_params, ea_stream, device_free_trace, batched

ARCHS = ("eegnex", "ifnet")
SEEDS = range(12)
CORPORA = ["physionetmi", "cho2017", "lee2019_mi", "dreyer2023", "schirrmeister2017", "shin2017a", "shin2017b"]
C_LR = 0.01


def verify_trunks(ckpt_dir):
    """Raise unless every trunk is a pinned external m7r file with no PROTEUS training data."""
    pins = dict(reversed(m.groups()) for m in re.finditer(
        r"^([0-9a-f]{64})  runs/pretrain/((?:eegnex|ifnet)_ea-m7r-s\d+\.pt)$",
        (ROOT / "scripts/sealed_nxif_inputs.sha256").read_text(), re.M))
    out = {}
    for arch in ARCHS:
        for s in SEEDS:
            name = f"{arch}_ea-m7r-s{s}.pt"
            path = Path(ckpt_dir) / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if pins.get(name) != digest:
                raise ValueError(f"trunk {name} does not match the pinned external hash")
            meta = json.loads(torch.load(path, map_location="cpu", weights_only=False)["meta"])
            if (list(meta["corpora"]) != CORPORA or meta["align"] != "ea" or meta["arch"] != arch
                    or meta["seed"] != s or any("uda" in k.lower() or "proteus" in str(v).lower()
                                               for k, v in meta.items())):
                raise ValueError(f"trunk {name} metadata is not the external m7r recipe: {meta}")
            out[name] = digest
    return out


def main():
    args = family_args(__doc__, (0,))
    smoke = os.environ.get("CAP250_SMOKE") == "1"
    ckpt_dir = ASSETS / "runs/pretrain"
    if not torch.cuda.is_available() and not smoke:
        raise RuntimeError("teacher fit needs CUDA inside Slurm (set CAP250_SMOKE=1 to test on CPU)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pins = verify_trunks(ckpt_dir)
    data = load_source(args.people, args.first_session_only, args.data)
    X, y, stream = data["X"], data["y"], data["stream"]
    if smoke:
        X, y, stream = X[:600], y[:600], stream[:600]
    p = ea_params(X)
    H = ea_stream(X, stream, p)
    seeds = range(1) if smoke else SEEDS
    heads = Heads(ckpt_dir, seeds, ARCHS).eval()
    heads_dev = copy.deepcopy(heads).to(dev)
    Fh = batched(heads_dev, H, dev=dev)
    lr = LogisticRegression(C=C_LR, max_iter=3000, class_weight="balanced").fit(Fh, y)
    teacher = Teacher(heads, lr.coef_, lr.intercept_).eval()
    traced = device_free_trace(teacher, torch.from_numpy(H[:4]))
    traced.save(str(args.out / "teacher.pt"))
    loaded = torch.jit.load(str(args.out / "teacher.pt"))
    eager, tr256, tr1 = batched(teacher, H), batched(loaded, H), batched(loaded, H[:64], n=1)
    d256, d1 = float(np.abs(tr256 - eager).max()), float(np.abs(tr1 - eager[:64]).max())
    on_gpu = None
    if dev == "cuda":
        g = torch.jit.load(str(args.out / "teacher.pt"), map_location="cuda")
        with torch.no_grad():
            on_gpu = float(np.abs(g(torch.from_numpy(H[:256]).cuda()).double().cpu().numpy() - eager[:256]).max())
    if d256 > 1e-4 or d1 > 1e-4 or (on_gpu is not None and on_gpu > 1e-3):
        raise ValueError(f"teacher trace mismatch: {d256} {d1} {on_gpu}")
    np.savez(args.out / "weights.npz", **{f"net34_{k}": v for k, v in p.items()},
             net34_jit_file=np.array("teacher.pt"))
    diag = args.out / "diagnostics"
    diag.mkdir()
    (diag / "teacher.json").write_text(json.dumps(dict(
        smoke=smoke, n_rows=len(y), trunks=pins, C=C_LR, traced_vs_eager=d256, batch1_vs_eager=d1,
        cuda_vs_eager=on_gpu, in_sample_acc=float(np.mean(eager.argmax(1) == y))), indent=1))
    np.savez(diag / "lr.npz", coef=lr.coef_, intercept=lr.intercept_)


if __name__ == "__main__":
    main()
