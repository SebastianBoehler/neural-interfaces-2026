"""Reconstruct source checkpoints and verify exact legacy refit consumption."""

import argparse
import hashlib
import json

import numpy as np
import torch

from paired_refit_smoke import TARGETS, load_target
from refit_smoke import HERE, ROOT, TOL, changed, file_sha, refit_api, waveform_data


RELEASE = ROOT.parent / "ni26_friday_causal_rebuild_20261118/refit_full0_matrix_v2"
RELEASE_SHA = "841d35a4e3d3bcd9dba44e4f7b03ef896622d64fa922b5da9ad7afb367cc698c"
FAMILIES = ("fbraw", "fb", "sinc", "fbl")
DEVICES = {"fbraw": "RTX 2080 Ti", "fb": "RTX 2080 Ti", "sinc": "A100", "fbl": "A100"}


def restore(api, data, family, person, path, expected_sha):
    from deep import build

    if file_sha(path) != expected_sha:
        raise ValueError(f"source checkpoint identity differs for {family} {person}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["actor"], payload["seed"], payload["fold"]) != (family, 0, person):
        raise ValueError("checkpoint does not match the requested actor")
    held = data["subject"] == f"Dreyer2026Proteus/{person}"
    fit = np.flatnonzero(~held)
    model = api.make(family, data, 0)
    model.G, model.scale = payload["prior"].numpy(), float(payload["scale"])
    np.testing.assert_array_equal(model.C[fit].mean(0), model.G)
    if not np.isfinite(model.scale) or model.scale <= 0 or (data["y"] != -1).any():
        raise ValueError("invalid source scaling or unexpectedly observed target labels")
    n_chan, n_time = model._aligned(fit[:2], causal=False).shape[1:]
    model.net = build(model.arch, n_chan, n_time, model.pretrained, **model.net_kw).to(model.dev)
    model._prepare()
    model.net.load_state_dict(payload["net"], strict=True)
    wanted = set(payload["trainable"])
    available = dict(model.net.named_parameters())
    if not wanted or not wanted.issubset(available):
        raise ValueError("trained parameter scope differs")
    for key, param in available.items():
        param.requires_grad_(key in wanted)
    model.set_k(api.ARCHS[family][1])
    model.net.eval()
    if not model.n_sbn:
        raise ValueError("source SessionBN restoration is inactive")
    return model, fit, held


def old_meta(family):
    if file_sha(RELEASE / "source.sha256") != RELEASE_SHA:
        raise ValueError("archived-anchored producer manifest differs")
    path = RELEASE / "members" / f"{family}_s0.json"
    meta = json.loads(path.read_text())
    fixed = dict(member=family, seed=0, smoke=False, order="full0", rows=2655,
                 targets=TARGETS, held_labels_opened=False)
    if any(meta[k] != v for k, v in fixed.items()) or len(meta["fit_states"]) != 10:
        raise ValueError("missing complete fit receipt")
    if file_sha(path.with_suffix(".npz")) != meta["output_sha256"]:
        raise ValueError("refit reference bytes changed")
    return meta, path.with_suffix(".npz")


def smoke(family):
    api, stream, actor_sha = refit_api(DEVICES[family])
    meta, reference_path = old_meta(family)
    person = "P01"
    entry = meta["fit_states"][person]
    data = waveform_data()
    path = RELEASE / "checkpoints" / f"{family}_s0_{person}.pt"
    model, fit, held = restore(api, data, family, person, path, entry["sha256"])
    if hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest() != entry["fit_rows_sha"]:
        raise ValueError("source mask differs from its archived-anchored receipt")
    with np.load(reference_path, allow_pickle=False) as bank:
        rows = bank["delivery"][held[bank["delivery"]]]
        expected = {name: bank[name][rows] for name in ("base", *TARGETS)}
    np.testing.assert_array_equal(np.sort(rows), np.flatnonzero(held))
    x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
    output = {}
    for arm in TARGETS:
        refit = stream(model, x, load_target(arm, rows, (len(rows), 58)), 0)
        output[arm] = changed(refit["r15c250"], expected[arm])
        base = changed(refit["base"], expected["base"])
        if any(v["changed"] or v["max_abs"] > TOL for v in (base, output[arm])):
            raise ValueError(f"source restoration differs: {family}, {arm}, {base}, {output[arm]}")
    receipt = HERE / f"restore_{family}_P01.json"
    with receipt.open("x") as handle:
        json.dump(dict(actor_sha256=actor_sha, fold=person, family=family, checkpoint_sha256=entry["sha256"],
                       held_labels_opened=False, base=base, refits=output), handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(f"PASS restored {family} P01", file_sha(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("fbraw", "fb", "fbl"))
    smoke(parser.parse_args().family)
