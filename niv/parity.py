"""Parity of the NI-V worker with the archived Friday exact-archive run: same ZIP, person G, visit 1 and 2, order 22411.
python parity.py NIV_PARITY.npz FRIDAY_person_5.json  -> prints decisions equal / total (GPU run-to-run differences allowed)."""
import json
import sys

import numpy as np

z = np.load(sys.argv[1], allow_pickle=False)
r = json.load(open(sys.argv[2]))["results"]
for v in (1, 2):
    mine = z[f"full|s22411|Scherer2015/G|{v}|labels"].astype(int)
    rows = z[f"full|s22411|Scherer2015/G|{v}|rows"]
    ref = r[f"Scherer2015/G|{v}|22411"]
    assert np.array_equal(rows, np.array(ref["rows"])), "delivery order differs"
    stock = np.array(ref["labels"][0])
    print(f"visit {v}: {int((mine == stock).sum())}/{len(mine)} decisions equal to archived stock run A; rows identical")
