"""Rebuild the 24 historical full0 refit members from paired regenerated targets."""

import argparse
import hashlib
import json

import numpy as np
import torch

from paired_refit_smoke import TARGETS, load_target
from refit_smoke import ARCHIVE, HERE, ROOT, TOL, changed, file_sha, refit_api, source_data


FAMILIES = ("fbraw", "fb", "sinc", "fbl")
PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
ROWS = 2655
GRID_SIZE = 18


def archived(family, seed, name):
    path = ARCHIVE / f"ttt18full0LPC_{family}_{name}-all_loso-s{seed}.npz"
    with np.load(path, allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank["index"], np.arange(ROWS))
        return bank["proba"]


def checkpoint(model, stem, person, family, seed, fit):
    fit_sha = hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest()
    path = HERE / "checkpoints" / f"{stem}_{person}.pt"
    with path.open("xb") as handle:
        torch.save(dict(net={name: value.detach().cpu() for name, value in model.net.state_dict().items()},
                        trainable=[name for name, p in model.net.named_parameters() if p.requires_grad],
                        prior=torch.from_numpy(model.G), scale=model.scale, fold=person, actor=family,
                        seed=seed, fit_rows_sha=fit_sha), handle)
    return {"sha256": file_sha(path), "fit_rows_sha": fit_sha, "n_fit": len(fit)}


def replay_member(family, seed, smoke):
    gpu_pins = json.loads((HERE / "refit_gpu.json").read_text())
    stem = f"{family}_s{seed}"
    gpu_name = gpu_pins[stem]
    api, stream, code_sha = refit_api(gpu_name)
    reference_base = archived(family, seed, "base")
    reference_ft = archived(family, seed, "r15c250") if smoke else None
    reference_target = api.chain_labels("all_loso") if smoke else None
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as bank:
        delivery = bank["full|0|all_loso"]
    arrays = {name: np.full((ROWS, 3), np.nan) for name in ("base", *TARGETS)}
    people = PEOPLE[:1] if smoke else PEOPLE
    states, history_max = {}, 0.0
    for person in people:
        data, fit, test = source_data(f"Dreyer2026Proteus/{person}")
        model = api.make(family, data, seed)
        model.fit(fit)
        model.set_k(api.ARCHS[family][1])
        if model.dev != "cuda" or model.n_sbn < 1 or not (data["y"][test] == -1).all():
            raise ValueError(f"{stem} {person} has the wrong source fit or path")
        session_key = data["stream"][test]
        for session in dict.fromkeys(session_key):
            rows = delivery[data["stream"][delivery] == session]
            np.testing.assert_array_equal(np.sort(rows), test[session_key == session])
            x = torch.from_numpy(model._aligned(rows, causal=True)).to(model.dev)
            with torch.no_grad():
                frozen = torch.softmax(model.net(x)[:, :3].double(), 1).cpu().numpy()
            stats = changed(frozen, reference_base[rows])
            if stats["changed"] or stats["max_abs"] > TOL:
                raise ValueError(f"{stem} {person} frozen archive parity failed: {stats}")
            history_max = max(history_max, stats["max_abs"])
            arrays["base"][rows] = frozen
            if smoke:
                target = np.stack([reference_target[int(row)] for row in rows])
                reverted = stream(model, x, target, seed)["r15c250"]
                check = changed(reverted, reference_ft[rows])
                if check["changed"] or check["max_abs"] > TOL:
                    raise ValueError(f"{stem} archived refit parity failed: {check}")
            for arm in TARGETS:
                target = load_target(arm, rows, (len(rows), 4 + 3 * GRID_SIZE))
                replay = stream(model, x, target, seed)["r15c250"]
                np.testing.assert_array_equal(replay[:20], frozen[:20])
                arrays[arm][rows] = replay
        if not smoke:
            states[person] = checkpoint(model, stem, person, family, seed, fit)
        print(f"PASS {stem} {person} {len(test)} held rows", flush=True)
    ids = np.flatnonzero(np.isfinite(arrays["base"]).all(1))
    if len(ids) != (190 if smoke else ROWS):
        raise ValueError("incomplete member bank")
    arrays = {name: value[ids] for name, value in arrays.items()}
    delta = changed(arrays["causal"], arrays["historical"])
    if delta["max_abs"] <= 1e-4:
        raise ValueError(f"{stem} regenerated targets did not change the actual refit")
    out = HERE / ("smoke" if smoke else "members") / f"{stem}.npz"
    with out.open("xb") as handle:
        np.savez_compressed(handle, index=ids, delivery=delivery[np.isin(delivery, ids)], **arrays)
    receipt = dict(member=family, seed=seed, smoke=smoke, order="full0", rows=len(ids), fit_states=states,
                   archived_frozen_max_abs=history_max, paired_refit_delta=delta, output_sha256=file_sha(out),
                   actor_sha256=code_sha, targets=TARGETS, gpu=torch.cuda.get_device_name(),
                   held_labels_opened=False)
    with out.with_suffix(".json").open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=int, choices=range(24))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    replay_member(FAMILIES[args.task % len(FAMILIES)], args.task // len(FAMILIES), args.smoke)
