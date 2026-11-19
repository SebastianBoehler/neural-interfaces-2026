"""Authenticate the corrected parent banks before using new predictions."""
import json
from pathlib import Path

import numpy as np
from bridge_owner import pin, sha

PARENT = Path("/mnt/lustre/work/gehler/gfh527/ni26_friday_c21_corrected_v2_20261119")
PINS = {
    "within0": ("c41f6df763d86dee454db348560fce1943620f6ed3a3a1ef8824d4c54ac38472",
                "109a572de0f062a62ddf69d80f847d5a107e33236b45f34d7984af303df6df45"),
    "full0": ("442ba868279c332e550bc4cffbd29f372c1e1afdf7fb8d6e86fe1a548fd6f331",
              "5b9da9b5ba865c4b0b6bcdd2251536a472884d561bda85cfb4e624358f3f5650"),
}


def compare_parent(order, delivery, selected, results):
    receipt, bank = PARENT / f"{order}.json", PARENT / f"{order}.npz"
    proof, array = PINS[order]
    pin(receipt, proof)
    pin(bank, array)
    record = json.loads(receipt.read_text())
    assert record["passed"] and record["rows"] == 2655 and not record["held_labels_opened"]
    assert record["output"]["sha256"] == sha(bank) and record["stock_bitwise"] and record["native_identity"]
    with np.load(bank, allow_pickle=False) as reference:
        np.testing.assert_array_equal(reference["delivery"], delivery)
        for arm in ("stock", "identity", "FBL2"):
            for stage, values in results[arm].items():
                np.testing.assert_array_equal(values[selected], reference[f"{arm}|{stage}"][selected])
    return record
