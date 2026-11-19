"""cap250 reference through the harness scorer (HELD-NEW only, one legacy order).

No cap250 LOPO fold ZIPs exist: cap250's held-new numbers come from the c25d replay banks (nine-person fits, frozen
banks + stage chain replayed on the delivery). This adapter reads the `stock|terminal` score vectors of the c25d full0 bank
(Galvani ni26_friday_c25d_held_v1_20261119/full0.npz, 2,655 rows, index = data row, shared fresh Sinc seed, pinned legacy
order "full0" = order id 0) and argmax-es them into the harness prediction format, so score.py/compare.py treat cap250 like
any team. NOT streamed here, no returning arm, order 0 is not 71000-71007: compare.py then averages each side over orders.
python reference_cap250.py BANK.npz RUN_DIR      (then: python score.py RUN_DIR)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402


def main(bank, run, arm="stock"):
    d = D.load()
    with np.load(bank, allow_pickle=False) as z:
        assert np.array_equal(z["index"], np.arange(len(d["y"])))
        labels = z[f"{arm}|terminal"].argmax(1).astype(np.int8)
        delivery = z["delivery"]
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    for p in D.PEOPLE:
        res = {}
        for s in sorted(set(d["session"][d["person"] == p])):
            rows = delivery[np.isin(delivery, np.flatnonzero(d["stream"] == f"{p}|{s}"))]
            res[f"rows_0_{s}"], res[f"labels_0_{s}"] = rows.astype(np.int64), labels[rows]
        np.savez(run / f"pred_new_{p}.npz", **res)
    (run / "jobs.tsv").write_text("".join(f"new\t{p}\t-\t-\tall\n" for p in D.PEOPLE))


if __name__ == "__main__":
    main(*sys.argv[1:])
