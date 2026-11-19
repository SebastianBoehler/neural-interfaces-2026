"""Check the fixed RTX-trained FB raw state against the original A100 runs0 anchor."""

import json

import numpy as np
import torch

from refit_smoke import HERE, ROOT, TOL, file_sha, waveform_data
from restore_refit import RELEASE
from within0_frozen_probe import rows_sha
from within0_warm import actor
from member_inputs import ORDERS, pinned_delivery


def main():
    order, person = "runs0", "P01"
    api, _, _ = actor("A100", order, read_legacy=False)
    d = waveform_data()
    p = pinned_delivery(ROOT / "runs/shuffle_perms.npz")[ORDERS.index(order)]
    held = d["subject"] == f"Dreyer2026Proteus/{person}"
    src = np.flatnonzero(~held)
    rows, seq = np.flatnonzero(held), p[held[p]]
    reset = p[d["stream"][p] == "Dreyer2026Proteus/P02|1"][:80]
    if len(rows) != 190 or len(reset) != 80 or np.array_equal(rows, seq):
        raise ValueError("unexpected real canary scope")
    from fresh_sinc import evaluate, replay, state_hash
    from restore_refit import restore
    from within0_frozen_probe import old_base
    from refit_smoke import changed

    spec = json.loads((RELEASE / "members/fbraw_s0.json").read_text())["fit_states"][person]
    path = RELEASE / "checkpoints/fbraw_s0_P01.pt"
    m, fit, mask = restore(api, d, "fbraw", person, path, spec["sha256"])
    np.testing.assert_array_equal(fit, src)
    np.testing.assert_array_equal(mask, held)
    if rows_sha(fit) != spec["fit_rows_sha"]:
        raise ValueError("the canary state includes the wrong training rows")
    state = state_hash(m.net)
    values, prefix = replay(m, d, rows, seq)
    partial = evaluate(m, seq[:80])
    evaluate(m, reset)
    np.testing.assert_array_equal(evaluate(m, seq[:80]), partial)
    np.testing.assert_array_equal(m._aligned(seq, True)[:1], m._aligned(seq[:1], True))
    ep = changed(partial, values[np.searchsorted(rows, seq[:80])])
    if ep["changed"] or ep["max_abs"] > TOL or state_hash(m.net) != state:
        raise ValueError("FB raw current-device prefix failed")
    anchor = changed(values, old_base("fbraw", rows, order))
    eligible = anchor["changed"] == 0 and anchor["max_abs"] <= TOL
    proof = dict(order=order, person=person, rows=len(rows), gpu=torch.cuda.get_device_name(), eligible=eligible,
                 state=spec["sha256"], source=file_sha(HERE / "canary_source.sha256"), prefix=ep,
                 internal_prefix=prefix, exact_reset=True, held_labels_opened=False, anchor=anchor)
    out = HERE / "fbraw_canary.json"
    with out.open("x") as handle:
        json.dump(proof, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    out.chmod(0o444)
    print("PASS" if eligible else "REFUTED", proof, flush=True)
    if not eligible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
