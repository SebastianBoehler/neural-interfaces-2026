"""Independent S1C replay of the valid trained/eval fixed-feature endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from held import frozen_member
from replay_f13 import DATA_SHA, ORDER_SHA, ORDERS, ORIGIN_SHA, PEOPLE, origin_report, raw_inputs
from replay_f13 import row_sha, session_batches, sha
from skip_policy import KINDS, fixed_features


HERE = Path(__file__).resolve().parent
F14 = HERE.parent / "ni26_friday_s1c_training_20261118"
F14_BANK = "b012b681c8bac77a11a83ce07dd9e2b171ce4eb33ca322a175de24afdcaa7920"
NORMALIZERS = ("ordinary", "bounded")
Q_MODES = {kind: ("zero",) if kind == "zero" else ("on", "zero") for kind in KINDS}
ARMS = tuple(f"{kind}_{norm}_{q}" for kind in KINDS for norm in NORMALIZERS for q in Q_MODES[kind])
TOL = 2e-5


def planned_masks(batches, held, seed):
    generator = torch.Generator(device="cpu").manual_seed(15015 + 100 * int(held[1:]) + seed)
    digest = hashlib.sha256()
    for rows in batches:
        selected = torch.randperm(len(rows), generator=generator)[:len(rows) // 4].numpy()
        mask = np.zeros(len(rows), dtype=bool)
        mask[selected] = True
        digest.update(mask.tobytes())
    return digest.hexdigest()


def authenticate(held, seed, subject, person, stream, smoke):
    import model322

    fit = np.flatnonzero(person != held)
    model322.SEED = seed
    epochs = 4 if smoke else 40
    batches = [batch for epoch in model322.batches(stream[fit], epochs) for batch in epoch]
    count = sum(map(len, batches))
    group_hash = hashlib.sha256(b"".join(fit[b].astype("<i8").tobytes() for b in batches)).hexdigest()
    fixed = dict(held=held, seed=seed, epochs=epochs, dataset=DATA_SHA, origin=ORIGIN_SHA,
                 actor=sha(HERE / "actor.sha256"), runtime=sha(F14 / "runtime.lock"), batches=group_hash,
                 source_rows=row_sha(fit), source_people=sorted(set(subject[fit])), held_labels_opened=False,
                 steps=len(batches), parameters=347903, train_rows_seen=count, planned_mask_rows=count // 4,
                 mask_plan=planned_masks(batches, held, seed), mask_rng_isolated=True)
    pairs = []
    base_tag = f"{'smoke_' if smoke else ''}original_{seed}_{held}"
    base = (F14 / "checkpoints" / f"{base_tag}.json")
    f14 = json.loads(base.read_text())
    if sha(base.with_suffix(".pt")) != f14["checkpoint"]:
        raise ValueError("original F14 control checkpoint changed")
    for kind in KINDS:
        tag = f"{'smoke_' if smoke else ''}{kind}_{seed}_{held}"
        path = HERE / "checkpoints" / f"{tag}.pt"
        meta = json.loads(path.with_suffix(".json").read_text())
        expected = dict(fixed, kind=kind, checkpoint=sha(path), initial=f14["initial"], scale=f14["scale"],
                        f14_original_state=f14["state"], active_mask_rows=count // 4 if kind == "dropout" else 0)
        if (any(meta[k] != v for k, v in expected.items()) or meta["device"] != f14["device"] or
                meta["braindecode"] != "1.8.1" or meta["torch"] != f14["torch"]):
            raise ValueError(f"source-only matched fit differs for {tag}")
        if ((meta["max_q_gradient"] == 0) != (kind == "zero") or not np.isfinite(meta["max_q_gradient"])
                or (meta["q_gradient_active_steps"] == 0) != (kind == "zero")):
            raise ValueError(f"wrong original fixed-branch gradient for {tag}")
        if kind == "original" and meta["state"] != f14["state"]:
            raise ValueError("q-on original source state differs from the same-platform reference")
        pairs.append((path, meta))
    if len({m["state"] for _, m in pairs}) != len(KINDS):
        raise ValueError("duplicate source model states")
    return pairs


def probability(model, x, qs):
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    z = model.representation(x)
    q = fixed_features(model, x)
    logits = torch.stack([model.task(torch.cat((z, q if mode == "on" else torch.zeros_like(q)), 1)) for mode in qs])
    return torch.softmax(logits.double(), dim=2).cpu().numpy()


@torch.no_grad()
def replay(model, layer, kind, x, stream, test, perms, scale):
    from encoder312 import fingerprint
    from modes import probabilities

    qs = Q_MODES[kind]
    before, forward = fingerprint(model), layer.forward
    result = np.full((2, len(qs), len(ORDERS), len(test), 3), np.nan)
    counts = []
    bounded = MethodType(lambda self, inp: normalize(self, inp, counts), layer)
    first, active, max_diff, direct = 0, 0, 0.0, False
    for order, pos, xx, other in session_batches(x, stream, test, scale, perms):
        ordinary = probability(model, xx, qs)
        prefix = probability(model, xx[:10], qs)
        if not direct:
            for q, val in zip(qs, prefix, strict=True):
                model.skip = q == "on"
                if not np.array_equal(val, probabilities(model, xx[:10], "cpu")):
                    raise ValueError("shared representation path differs from the native q control")
            model.skip = kind != "zero"
            direct = True
        if not np.allclose(ordinary[:, :prefix.shape[1]], prefix, atol=TOL, rtol=TOL):
            raise ValueError("future inputs changed an ordinary prefix")
        layer.forward = bounded
        counts.clear()
        controlled = probability(model, xx, qs)
        if len(counts) != 1 or not np.array_equal(ordinary[:, 0], controlled[:, 0]):
            raise ValueError("normalizers changed the actual first window at fixed q")
        active += int(counts[0])
        prefix = probability(model, xx[:10], qs)
        max_diff = max(max_diff, float(np.abs(controlled[:, :prefix.shape[1]] - prefix).max()))
        if not np.allclose(controlled[:, :prefix.shape[1]], prefix, atol=TOL, rtol=TOL):
            raise ValueError("future inputs changed a bounded prefix")
        if len(other):
            probability(model, other, qs)
            if not np.array_equal(probability(model, xx[:10], qs), prefix):
                raise ValueError("bounded inference retained state across sessions")
        layer.forward = forward
        result[:, :, ORDERS.index(order), pos] = np.stack((ordinary, controlled))
        first += 1
    if fingerprint(model) != before or not np.isfinite(result).all() or not np.allclose(result.sum(4), 1):
        raise ValueError("held prediction changed frozen weights or missed canonical rows")
    predictions = {f"{kind}_{norm}_{q}": result[i, j].argmax(2).astype(np.int8)
                   for i, norm in enumerate(NORMALIZERS) for j, q in enumerate(qs)}
    return predictions, dict(first_windows_exact=first, prefix_max_abs=max_diff, active_eval_clips=active)


def run(held, seed, smoke):
    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    test = np.flatnonzero(person == held)
    predictions, evidence = {}, {}
    for path, meta in authenticate(held, seed, subject, person, stream, smoke):
        kind = meta["kind"]
        values, guards = replay(*frozen_member(path, meta), kind, x, stream, test, perms, meta["scale"])
        predictions.update(values)
        evidence[kind] = dict(guards, checkpoint=sha(path), receipt=sha(path.with_suffix(".json")),
                              state=meta["state"], max_q_gradient=meta["max_q_gradient"],
                              active_mask_rows=meta["active_mask_rows"])
    if tuple(predictions) != ARMS or not any(g["active_eval_clips"] for g in evidence.values()):
        raise ValueError("held replay included an invalid q endpoint or no bounded activity")
    receipt = dict(held=held, seed=seed, dataset=DATA_SHA, origin=ORIGIN_SHA, order_sha=ORDER_SHA, orders=list(ORDERS),
                   held_labels_opened=False, scorer=sha(HERE / "score.sha256"), actor=sha(HERE / "actor.sha256"),
                   evidence=evidence)
    if not smoke:
        if sha(F14 / "before_labels.sha256") != F14_BANK:
            raise ValueError("same-platform F14 blind bank identity differs")
        with np.load(F14 / "folds" / f"{held}_s{seed}.npz", allow_pickle=False) as base:
            if not np.array_equal(base["index"], test) or any(not np.array_equal(
                    predictions[f"original_{mode}_on"], base[f"original_{mode}"]) for mode in NORMALIZERS):
                raise ValueError("original q-on held decision bank changed from authentic F14")
        output = HERE / "folds" / f"{held}_s{seed}.npz"
        with output.open("xb") as handle:
            np.savez_compressed(handle, index=test, metadata=json.dumps(receipt, sort_keys=True), **predictions)
        print(f"{sha(output)}  {output.name}", flush=True)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print("PASS F15 independent held inference", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("held", choices=PEOPLE)
    p.add_argument("seed", choices=(0, 1), type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(a.held, a.seed, a.smoke)
