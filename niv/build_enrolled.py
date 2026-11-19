"""Build the enrolled packages of the returning arm: for each Scherer person k, SRC with visit 1 of k enrolled.

python build_enrolled.py SRC.zip OUT_DIR TAG    (runs in the worker image; writes OUT_DIR/TAG_k{0..8}.zip)
Visit 1 is taken in acquisition order (run, onset), the way the sealed package holds a public person's session.
Labels of visit 1 are used for enrolment only; visit 2 is never opened here.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/official/tracks/bci_decoding")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import enrol  # noqa: E402
import orders  # noqa: E402


def main(src, out_dir, tag):
    with np.load("/external.npz", allow_pickle=False) as z:
        X, y, subject, session, onset = z["X"], z["y"], z["subject"], z["session"], z["onset"]
    run = orders.runs_from_manifest("/rows.tsv.gz", subject, session)
    for k, who in enumerate(sorted(set(subject))):
        rows = np.flatnonzero((subject == who) & (session == "1"))
        rows = rows[np.lexsort((onset[rows], run[rows]))]
        assert len(rows) in (105, 120) and set(y[rows]) == {0, 1, 2}
        n = enrol.enrol(src, f"{out_dir}/{tag}_k{k}.zip", X[rows], y[rows], f"S{who[-1]}", os.environ.get("NIV_OWN_S", "stock"))
        print("enrolled", who, "->", n, "people", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
