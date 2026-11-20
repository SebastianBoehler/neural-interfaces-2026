"""Fit and export an F16e/F32e hard waveform actor on the permitted PROTEUS rows."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ASSETS, ROOT, family_args, file_digest, load_source

RELEASE_HASH = "88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5"


def native_recipe():
    path = Path(os.environ.get("CAP250_HARD_SOURCE", ASSETS / "ni26_s323_normsplit_20261117")).resolve()
    manifest = path / "source323.sha256"
    if file_digest(manifest) != RELEASE_HASH:
        raise ValueError(f"wrong pinned §323 hard source: {path}")
    pins = {name: digest for digest, name in (line.split() for line in manifest.read_text().splitlines())}
    for name in ("model322.py", "encoder312.py"):
        if file_digest(path / name) != pins[name]:
            raise ValueError(f"the training recipe changed: {path / name}")
    prefix = "experiments/"
    for name, digest in pins.items():
        if name.startswith(prefix) and "/._" not in name and file_digest(path / name) != digest:
            raise ValueError(f"the training dependency changed: {path / name}")
    sys.path[:0] = [str(path), str(path / "experiments"),
                    str(ROOT / "2026-competition/tracks/bci_decoding"),
                    str(ROOT / "friday/cap250_raw_swap/model")]
    import encoder312
    import model322
    import session_bn
    from incremental_model import export_actor

    if Path(model322.__file__).resolve() != path / "model322.py" or Path(encoder312.__file__).resolve() != path / "encoder312.py":
        raise RuntimeError("wrong hard recipe module import")
    return encoder312, model322, session_bn.SessionBN, export_actor


def fit(args):
    import braindecode
    if not torch.cuda.is_available() or braindecode.__version__ != "1.8.1":
        raise RuntimeError(f"hard fit requires Slurm CUDA and Braindecode 1.8.1 "
                           f"(CUDA available: {torch.cuda.is_available()}, "
                           f"torch: {torch.__version__}, braindecode: {braindecode.__version__})")
    torch.set_num_threads(2)
    encoder, recipe, norm_class, export = native_recipe()
    data = load_source(args.people, args.first_session_only, args.data)
    recipe.SEED = args.seed
    epochs = 4 if os.environ.get("CAP250_SMOKE") == "1" else 40
    x, y, scale, mu, sd, weights, batches = recipe.setup(data, np.arange(len(data["y"])), "cuda", epochs=epochs)
    model = recipe.Factor("session", False, "cuda")
    (norm,) = [mod for mod in model.body.modules() if type(mod) is norm_class]
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if (norm.m, norm.k, norm.k_train, norm.causal, params) != (32, 20., 20., False, 347903):
        raise ValueError("hard fit architecture or training normalization changed")
    initial = encoder.fingerprint(model)
    zero_columns = []

    def zero_gradient(gradient):
        zero_columns.append(gradient[:, -287:].count_nonzero())

    hook = model.task.weight.register_hook(zero_gradient)
    result = recipe.train(model, x, y, mu, sd, weights, batches)
    hook.remove()
    final = encoder.fingerprint(model)
    if final == initial or len(zero_columns) != result["steps"] or torch.stack(zero_columns).any():
        raise ValueError("inactive hard fit or unexpected auxiliary-feature gradient")

    x = torch.from_numpy(data["X"][:17])
    actor = export(model.cpu(), norm, scale, x[:1])
    path = args.out / f"hard_s1c_s{args.seed}.pt"
    actor.save(str(path))
    loaded = torch.jit.load(str(path), map_location="cpu").eval()

    @torch.no_grad()
    def prefix(module, length):
        if int(module.reset()) != 0:
            raise ValueError("the actor failed to reset")
        return torch.cat([module(row[None]) for row in x[:length]])

    with torch.jit.optimized_execution(False):
        p = prefix(loaded, len(x))
        if p.dtype != torch.float64 or p.shape != (len(x), 3) or (p < 0).any():
            raise ValueError("unexpected hard actor output")
        torch.testing.assert_close(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-12, rtol=0)
        torch.testing.assert_close(prefix(loaded, 8), p[:8], atol=0, rtol=0)
        torch.testing.assert_close(prefix(actor, len(x)), p, atol=0, rtol=0)
    diagnostics = args.out / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "fit.json").write_text(json.dumps(dict(result, epochs=epochs, scale=scale,
        initial=initial, final=final, gpu=torch.cuda.get_device_name(), source_recipe=RELEASE_HASH), indent=2) + "\n")


if __name__ == "__main__":
    fit(family_args(__doc__, (0, 1)))
