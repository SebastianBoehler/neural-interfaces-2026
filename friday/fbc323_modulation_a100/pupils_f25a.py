"""Authenticate source exclusions and derive blind bounded CPU posteriors."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from encoder312 import fingerprint
from held_distill import raw_probability
from held_modulation import frozen_member
from modulation import ACTIVE, KINDS, MAX_RATIO, MAX_RESERVED, architecture
from pupils import do_order
from recipe_f25a import BD, CANARY_SHA, CPU_CUDA, F25_ACTOR, GPU, TORCH, WORK, anchors, canary, historical, tag
from replay_f13 import ORDER_SHA, ORDERS as ORIGINAL_ORDERS, origin_report, raw_inputs


ORDERS = ("full0", "within0")
EFFECTS = ("band_grad", "tap_grad", "identity_delta", "dilation_delta", "cross_band", "edge_tap")


def group(held, seed, smoke, subject, visit):
    rows, stream, people = split(subject, visit, held)
    reference, batches, metadata = historical(held, seed, smoke, rows, stream, people)
    controls, fold = anchors(held, seed, smoke)
    hard = controls["bare"][1]
    flags = canary()
    flat = [b for ep in batches for b in ep]
    expected = dict(held=held, seed=seed, epochs=reference["epochs"], scale=hard["scale"], initial=hard["initial"],
                    data=DATA_SHA, origin=ORIGIN_SHA, runtime=RUNTIME, source_rows=row_sha(rows), source_people=people,
                    batches=reference["batches"], historical_metadata=metadata, h100_state=reference["state"],
                    h100_weights_loaded=False, held_labels_opened=False, actor=sha(WORK / "actor.sha256"),
                    f25=F25_ACTOR, canary=CANARY_SHA, device=GPU, torch=TORCH, braindecode=BD,
                    **{k:flags[k] for k in CPU_CUDA}, steps=reference["steps"], bare_checkpoint=hard["checkpoint"],
                    bare_state=hard["state"], references={k:v[2] for k,v in controls.items()},
                    train_rows_seen=sum(map(len, flat)), diagnostic_rows=row_sha(rows[flat[0][:32]]),
                    rng_isolated=True, zero_identity=True, q_gradients_nonzero=0)
    cases = {}
    for kind in KINDS:
        path = WORK / "checkpoints" / f"{kind}_{tag(held, seed, smoke)}.pt"
        rec = path.with_suffix(".json")
        m = json.loads(rec.read_text())
        signed = dict(expected, kind=kind, checkpoint=sha(path), **architecture(kind))
        if (path.stat().st_mode & 0o222 or rec.stat().st_mode & 0o222
                or any(m[k] != v for k,v in signed.items()) or not 0 < m["peak_reserved"] <= MAX_RESERVED
                or not np.isfinite([m[k] for k in ("first_loss", "last_loss", "max_gradient")]).all()
                or m["max_gradient"] <= 0 or m["state"] in (m["initial"], m["h100_state"])):
            raise ValueError(f"authenticated A100 source path differs for {kind}/{held}/s{seed}")
        if kind == "hard":
            if m["state"] != hard["state"] or m["checkpoint"] != hard["checkpoint"]:
                raise ValueError("independent hard A100 bytes or initial/fit scope differ")
        else:
            delta = [m[k] for k in EFFECTS]
            if (not np.isfinite(delta).all() or min(delta) <= ACTIVE
                    or not .75 <= m["gain_min"] < m["gain_max"] <= 1.25):
                raise ValueError("original cross-band/temporal state or gain inactive")
        cases[kind] = path, m
    if len({m["state"] for _,m in cases.values()}) != len(KINDS):
        raise ValueError("independently trained variants share state")
    if smoke and any(m["hard_ms"] != cases["hard"][1]["warm_ms"] or not 0 < m["warm_ms"] <= m["hard_ms"] * MAX_RATIO
                     for _,m in cases.values()):
        raise ValueError("source smoke exceeded runtime bounds")
    return cases, controls["bare"][:2], fold


@torch.no_grad()
def probabilities(member, x, stream, rows, perms):
    path, meta = member
    model, layer = frozen_member(path, meta)
    model.skip = False
    output = {}
    exact = 0
    for order in ORDERS:
        p, prefix, clips = do_order(model, layer, x, stream, rows, perms[order], meta["scale"])
        for name in dict.fromkeys(stream[rows]):
            delivered = perms[order][stream[perms[order]] == name]
            original = raw_probability(model, x[delivered] / meta["scale"])
            np.testing.assert_array_equal(original[0], p[np.searchsorted(rows, delivered[0])])
            exact += 1
        output[order] = p
        output[f"{order}|proof"] = np.array([prefix, clips])
    if fingerprint(model) != meta["state"]:
        raise ValueError("held model tensors changed")
    return output, exact


def run(held, seed, smoke):
    torch.set_num_threads(2)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    visit = np.array([name.rsplit("|",1)[-1] for name in stream])
    rows = np.flatnonzero(person == held)
    cases, bare, fold = group(held, seed, smoke, subject, visit)
    control, _ = probabilities(bare, x, stream, rows, perms)
    values, proofs = {}, {}
    for kind, member in cases.items():
        p, exact = probabilities(member, x, stream, rows, perms)
        if not exact == len(ORDERS)*len(set(stream[rows])):
            raise ValueError("incomplete independent first-window identity")
        for key,v in p.items():
            values[f"{kind}|{key}"] = v
        path, meta = member
        proofs[kind] = dict(checkpoint=meta["checkpoint"], state=meta["state"], receipt=sha(path.with_suffix(".json")),
                            exact_first_windows=exact, exact_interleaved_resets=exact)
    for order in ORDERS:
        np.testing.assert_array_equal(control[order], values[f"hard|{order}"])
        np.testing.assert_array_equal(control[f"{order}|proof"], values[f"hard|{order}|proof"])
    if fold is not None:
        with np.load(fold, allow_pickle=False) as prior:
            np.testing.assert_array_equal(prior["index"], rows)
            for order in ORDERS:
                k = ORIGINAL_ORDERS.index(order)
                np.testing.assert_array_equal(values[f"hard|{order}"].argmax(1), prior["hard_bounded"][k])
    destination = WORK / ("smoke" if smoke else "pupils") / f"{held}_s{seed}.npz"
    metadata = dict(person=held, seed=seed, data=DATA_SHA, order_sha=ORDER_SHA, f25=F25_ACTOR, device=GPU,
                    weights=sha(WORK / "actor.sha256"), source=sha(WORK / "pupil.sha256"),
                    sealed_hard_fold=sha(fold) if fold is not None else None,
                    bare_equals_hard=True, held_labels_opened=False, arms=proofs)
    with destination.open("xb") as out:
        np.savez_compressed(out, index=rows, metadata=json.dumps(metadata, sort_keys=True), **values)
    sha_file = destination.with_suffix(".sha256")
    with sha_file.open("x") as out:
        out.write(sha(destination) + "\n")
    destination.chmod(0o444)
    sha_file.chmod(0o444)
    print("PASS F25a held pupils", held, seed, sha(destination), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held", choices=PEOPLE)
    parser.add_argument("seed", type=int, choices=(0, 1))
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    run(a.held, a.seed, a.smoke)
