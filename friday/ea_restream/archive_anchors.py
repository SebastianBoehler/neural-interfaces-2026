"""Authenticate the previously qualified FBCNet recordings and weight states."""
import json
from pathlib import Path

import numpy as np


BASE = Path("/mnt/lustre/work/gehler/gfh527")
PARENTS = {
    ("P01", 0): (BASE / "ni26_friday_ea_restream_20261118", "fbcnet_ea_tf_recording.json"),
    ("P10", 0): (BASE / "ni26_friday_ea_anchors_20261118", "fbcnet_ea_tf_P10_seed0_recording.json"),
    ("P01", 1): (BASE / "ni26_friday_ea_anchors_20261118", "fbcnet_ea_tf_P01_seed1_recording.json"),
}


def load_anchor(held, seed, receipt, index, full0):
    from probe import digest

    parent = PARENTS.get((held, seed))
    if parent is None:
        return None, None, None
    directory, name = parent
    original = json.loads((directory / name).read_text())
    for key in ("cfg", "fold", "seed", "source", "archive_manifest", "fit_rows_sha", "held_rows_sha", "prior_sha"):
        if receipt[key] != original[key]:
            raise ValueError(f"previously qualified anchor disagrees on {key}")
    if not original["parity"] or original["recording_max_abs"] != 0 or original["held_labels_opened"]:
        raise ValueError("anchor lacks bit-exact recording proof")
    stem = f"fbcnet_ea_tf_{held}_seed{seed}"
    encoded = directory / f"{stem}_full0.npz"
    if digest(encoded) != original["prediction_sha"]:
        raise ValueError("previously qualified anchor bank changed")
    with np.load(encoded, allow_pickle=False) as z:
        if not np.array_equal(z["index"], index) or not np.array_equal(z["delivery"], full0):
            raise ValueError("previously qualified anchor rows changed")
        proba = z["proba"]
    checkpoint = None
    if (held, seed) != ("P01", 0):
        checkpoint = directory / f"{stem}_reconstructed.pt"
        if digest(checkpoint) != original["reconstructed_checkpoint_sha"]:
            raise ValueError("previously qualified anchor checkpoint changed")
    return original, proba, checkpoint


def restore(actor, checkpoint, receipt, prior):
    import deep
    import torch

    with torch.random.fork_rng(devices=[]):
        model = deep.build(actor.arch, 41, 480, actor.pretrained, **actor.net_kw)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = dict(cfg=receipt["cfg"], seed=receipt["seed"], held=receipt["fold"], source=receipt["source"],
                    fit_rows_sha=receipt["fit_rows_sha"])
    if any(payload[key] != value for key, value in expected.items()) or not np.array_equal(payload["prior"], prior):
        raise ValueError("anchor checkpoint belongs to a different source fold")
    model.load_state_dict(payload["state"], strict=True)
    for weight in model.spectral_filtering.parameters():
        weight.requires_grad_(True)
    actor.net = model.to(actor.dev).eval()
