"""Replay the missing returning warm families with common fresh source actors."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from fbl_refit_inputs import DAYS, FOLD, ROOT, SB, WHO, arms
from peer_refit import warm, write
from returning_assets import asset
from returning_bank import input_class
from returning_owner import BASE, digest, sha


NATIVE = BASE / "ni26_js321_20261117_returning_banks_complete1/native_banks_complete.json"
NATIVE_SHA = "272f5cf50de59de52ba114ef8204532ef00cb5bced4074238f9c6ff73296d523"
FAMILIES = ("fbraw", "sinc")


def frozen_reference(family, held):
    if sha(NATIVE) != NATIVE_SHA:
        raise ValueError("returning bank collector digest changed")
    bank = json.loads(NATIVE.read_text())
    if bank["status"] != "all576_native_returning_banks_complete_NO_ACCURACY":
        raise ValueError("returning bank collector is incomplete")
    found = [row for row in bank["records"]
             if (row["fold"], row["seed"], row["name"]) == (FOLD, 0, f"full0_{family}.npz")]
    if len(found) != 1:
        raise ValueError("missing unique full0 native actor")
    path = Path(found[0]["path"])
    if sha(path) != found[0]["sha256"]:
        raise ValueError("frozen actor reference changed")
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["index"], held):
            raise ValueError("frozen actor reference rows differ")
        return data["proba"], found[0]["sha256"]


@torch.no_grad()
def read_trace(model, x):
    SB.reset_stream(model)
    frames = torch.from_numpy(x)
    logits = torch.cat([model(row[None]) for row in frames])
    return torch.softmax(logits[:, :3].double(), dim=1).numpy()


def frozen(model, X, held, ref, inputs):
    checks = {}
    for visit, other in (("2", "3"), ("3", "2")):
        rows = inputs[visit]["s281_archive"]["index"]
        indices = np.searchsorted(held, rows)
        if not np.array_equal(held[indices], rows):
            raise ValueError("unverified visit rows")
        reference = ref[indices]
        actual = read_trace(model, X[visit])
        short = read_trace(model, X[visit][:29])
        read_trace(model, X[other][:29])
        repeat = read_trace(model, X[visit][:29])
        if not np.array_equal(short, actual[:29]) or not np.array_equal(short, repeat):
            raise ValueError("20-window refit prefix or session reset differs")
        sorted_ref = np.sort(reference, axis=1)
        margin = sorted_ref[:, -1] - sorted_ref[:, -2]
        changed = actual.argmax(1) != reference.argmax(1)
        err = float(np.abs(actual - reference).max())
        if (not np.isfinite(actual).all() or not np.allclose(actual.sum(1), 1)
                or np.any(changed & (margin > 2e-5)) or err > 1e-4):
            raise ValueError(f"frozen fresh actor differs on visit {visit}: {err:.9g}, {changed.sum()} labels")
        checks[visit] = dict(rows=len(rows), max_abs=err, changed=int(changed.sum()),
                             exact_prefix=True, exact_reset=True)
    return checks


def main(family):
    if family not in FAMILIES:
        raise ValueError("expected missing raw or Sinc family")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as data:
        d = {key: data[key] for key in ("X", "subject", "session", "split")}
    d["stream"] = np.char.add(np.char.add(d["subject"], "|"), d["session"])
    held_mask = (d["subject"] == WHO) & (d["session"] != "1")
    fit, held = np.flatnonzero(~held_mask), np.flatnonzero(held_mask)
    own = np.flatnonzero(d["stream"] == WHO + "|1")
    checkpoint, trace, proof = asset(family, FOLD, 0)
    if (len(fit), len(held), len(own)) != (2270, 385, 192) or not np.isin(own, fit).all():
        raise ValueError("unexpected returning source and held split")
    if ((proof["train_count"], proof["held_count"]) != (len(fit), len(held))
            or proof["train_rows_sha256"] != digest(fit.astype("<i8"))
            or proof["held_rows_sha256"] != digest(held.astype("<i8"))):
        raise ValueError("actor checkpoint source and held rows differ")
    checked_before = (sha(checkpoint), sha(trace))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameters = dict(align=family != "fbraw", k=5., G=ck["G"], scale=ck["scale"], shrink=.001)
    InputStream = input_class()
    inputs = arms(d)
    X = {}
    for visit in DAYS:
        rows = inputs[visit]["s281_archive"]["index"]
        stream = InputStream(parameters)
        X[visit] = np.stack([stream(row) for row in d["X"][rows]])
        check = InputStream(parameters)
        if not np.array_equal(X[visit][:29], np.stack([check(row) for row in d["X"][rows[:29]]])):
            raise ValueError("packaged input path changed its causal prefix")
    model = torch.jit.load(str(trace), map_location="cpu").eval()
    if any(torch.count_nonzero(b).item() for n, b in model.named_buffers() if n.rsplit(".", 1)[-1] in SB.STATE):
        raise ValueError("actor checkpoint has pre-existing session state")
    ref, ref_sha = frozen_reference(family, held)
    frozen_audit = frozen(model, X, held, ref, inputs)
    write("frozen.json", frozen_audit)
    result = warm(family, trace.read_bytes(), X, inputs)
    if (sha(checkpoint), sha(trace)) != checked_before:
        raise ValueError("authenticated source assets changed")
    report = dict(family=family, same_fresh_actor=True, historical_actor_parity=False, no_gold=True,
                  source_rows=digest(fit.astype("<i8")), held_rows=digest(held.astype("<i8")),
                  checkpoint=checked_before[0], trace=checked_before[1], frozen_reference=ref_sha,
                  frozen=frozen_audit, warm=result, source=sha(Path("../fresh_warm_source.sha256")))
    write("fresh_warm.json", report)
    print(json.dumps(dict(family=family, frozen=frozen_audit, warm=result["warm"],
                          weights_changed=result["weights_changed"], bank=result["warm_sha256"],
                          receipt=sha(Path("fresh_warm.json"))), sort_keys=True), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
