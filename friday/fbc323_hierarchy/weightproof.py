"""Check actual waveform and head parameters against the authenticated initialization."""
import json

import numpy as np
import torch

import model322
from anchors import PEOPLE, sha
from encoder312 import fingerprint
from held_hierarchy import ACTOR, HERE, paired
from replay_f13 import DATA, origin_report
from tree_head import install


def run():
    torch.set_num_threads(1)
    origin_report()
    with np.load(DATA, allow_pickle=False) as data:
        subject, visit = data["subject"], data["session"]
    proof = {}
    for seed in (0, 1):
        model322.SEED = seed
        initial = model322.Factor("session", False, "cpu")
        flat = fingerprint(initial)
        install(initial)
        hierarchy = fingerprint(initial)
        weights = {k: v.detach().clone() for k, v in initial.body.named_parameters()}
        head = initial.task.weight.detach().clone()
        for held in PEOPLE:
            for kind, (path, meta) in paired(held, seed, False, subject, visit).items():
                if not kind.startswith("hierarchy"):
                    continue
                if (meta["flat_initial"], meta["initial"]) != (flat, hierarchy):
                    raise ValueError("CPU initial tensors do not match the authenticated H100 source initialization")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                initial.load_state_dict(payload["state"], strict=True)
                body = {k: float((v.detach() - weights[k]).abs().max()) for k, v in initial.body.named_parameters()}
                move = (initial.task.weight.detach() - head).abs().amax(1)
                if (fingerprint(initial) != meta["state"] or not any(v > 0. for v in body.values()) or
                        not torch.all(move > 0) or not torch.equal(move, torch.tensor(
                            [meta["motor_weight_delta"], meta["conditional_weight_delta"]], dtype=move.dtype))):
                    raise ValueError("active head rows or actual body parameters did not move as recorded")
                proof[f"{held}_{seed}_{kind}"] = dict(checkpoint=sha(path), moved=sum(v > 0. for v in body.values()),
                                                      body_max=max(body.values()), head_move=move.tolist())
    if len(proof) != 40:
        raise ValueError("one of the forty fitted trees lacks an actual body-parameter check")
    output = HERE / "weightproof.json"
    with output.open("x") as out:
        json.dump(dict(actor=ACTOR, held_labels_opened=False, parameters=proof), out, sort_keys=True, indent=2,
                  allow_nan=False)
        out.write("\n")
    print(f"PASS F19 exact initial and actual learned parameters {sha(output)}", flush=True)


if __name__ == "__main__":
    run()
