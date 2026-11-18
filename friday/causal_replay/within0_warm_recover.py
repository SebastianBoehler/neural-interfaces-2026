"""Recover only the two FBLight fits whose warm archive parity failed."""

import argparse
import json
import os

import numpy as np
import torch

from fresh_sinc import state_hash
from member_inputs import ORDERS, pinned_delivery
from refit_smoke import HERE, ROOT, TOL, file_sha, source_data
from within0_frozen_probe import rows_sha
from within0_warm import CHAIN, FROZEN_SHA, FTREL, ORDER, RELAB, actor, reference, session


FAILED = HERE.parent / "within0_warm_v2/members/fbl.json"
FAILED_SHA = "d720d1fbf91bb85bdacbe1974a16c30140437322597b797aa7d6bd0121703e99"


def run(person):
    if file_sha(FAILED) != FAILED_SHA or os.environ["SLURMD_NODENAME"] != "galvani-cn114":
        raise ValueError("missing historical RTX recovery context")
    api, stream, targets = actor()
    data, fit, held_rows = source_data(f"Dreyer2026Proteus/{person}")
    model = api.make("fbl", data, 0)
    model.fit(fit)
    model.set_k(api.ARCHS["fbl"][1])
    if not model.n_sbn or model.net.training or (data["y"][held_rows] != -1).any():
        raise ValueError("source fit scope or execution differs")
    np.testing.assert_array_equal(model.C[fit].mean(0), model.G)
    state = state_hash(model.net)
    data["y"].fill(-1)
    delivery = pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORDERS.index(ORDER)]
    visits = {}
    old = reference("fbl")
    held = np.isin(delivery, held_rows)
    order = delivery[held]
    np.testing.assert_array_equal(np.sort(order), held_rows)
    if len(np.unique(data["subject"][fit])) != 9:
        raise ValueError("original fit-excluded scope differs")
    reset = delivery[data["stream"][delivery] == "Dreyer2026Proteus/P01|1"][:80]
    for name in dict.fromkeys(data["stream"][order]):
        rows = order[data["stream"][order] == name]
        visits[name] = session(stream, model, rows, targets, file_sha(RELAB), old, reset, False)
    if state_hash(model.net) != state:
        raise ValueError("historical refits mutated the recovered frozen fit")
    success = all(r[k]["changed"] == 0 and r[k]["max_abs"] <= TOL for r in visits.values() for k in ("warm", "base"))
    seed = HERE / f"fbl_s0_{person}.pt"
    if success:
        with seed.open("xb") as handle:
            torch.save(dict(actor="fbl", seed=0, fold=person, prior=torch.from_numpy(model.G), scale=model.scale,
                            trainable=[k for k, p in model.net.named_parameters() if p.requires_grad],
                            net={k: value.detach().cpu() for k, value in model.net.state_dict().items()}), handle)
        seed.chmod(0o444)
    rec = dict(order=ORDER, person=person, source=file_sha(HERE / "source.sha256"),
               original_failed_sha256=FAILED_SHA, actor_sha256=file_sha(FTREL), frozen_sha256=FROZEN_SHA,
               chain_sha256=file_sha(CHAIN), relab_sha256=file_sha(RELAB), held_labels_opened=False,
               fit_rows_sha=rows_sha(fit), fit_rows=len(fit), checkpoint_sha256=file_sha(seed) if success else None,
               qualified=success, visits=visits, gpu=torch.cuda.get_device_name(), node=os.environ["SLURMD_NODENAME"])
    path = seed.with_suffix(".json")
    with path.open("x") as handle:
        json.dump(rec, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    print("Recovered archive parity", success, file_sha(path))
    if not success:
        raise SystemExit("historical source recovery did not qualify")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=int, choices=(8, 10))
    run(f"P{parser.parse_args().task:02d}")
