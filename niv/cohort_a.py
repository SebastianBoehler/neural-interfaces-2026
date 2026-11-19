"""Cohort A (10 held public PROTEUS people, held-new role) from the corrected re-streamed replays on file.

python niv/cohort_a.py OUT.npz  A_full0.npz:full A_within0.npz:within ...
Each input is a friday/causal_replay matched_<order>.npz (labels of 'corrected|P0r' = v5.1b-equivalent incumbent and
'corrected|cap250' for all 2655 rows, every member re-streamed, EA corrected). Cells = person x session x interface
(official metric, % ), padded with NaN to 6 per person. Verified against the stored official score before use.
Only the orders that exist are included (full0, within0); the extension to more orders needs one 15-job chain per order.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import balanced_accuracy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def cells(labels, d):
    people = sorted(set(d["subject"]))
    out = np.full((len(people), 6), np.nan)
    for pi, p in enumerate(people):
        keys = sorted(set(zip(d["session"][d["subject"] == p], d["interface"][d["subject"] == p])))
        for ci, (s, i) in enumerate(keys):
            r = np.flatnonzero((d["subject"] == p) & (d["session"] == s) & (d["interface"] == i))
            out[pi, ci] = 100 * balanced_accuracy(d["y"][r], labels[r])
    return out


def main(out, *specs):
    d = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    d = {k: d[k] for k in ("y", "subject", "session", "interface")}
    res = {"incumbent": [], "cap250": [], "scheme": []}
    for spec in specs:
        path, scheme = spec.split(":")
        z = np.load(path)
        score = json.loads(Path(path.replace(".npz", "_score.json")).read_text())["absolute_balanced_accuracy_pct"]
        for arm, key in (("incumbent", "P0r"), ("cap250", "cap250")):
            c = cells(z[f"corrected|{key}"].astype(int), d)
            assert abs(np.nanmean(c) - score[f"corrected|{key}"]) < 1e-6, (path, key, np.nanmean(c), score)
            res[arm].append(c)
        res["scheme"].append(scheme)
        print(path, scheme, {a: round(float(np.nanmean(res[a][-1])), 4) for a in ("incumbent", "cap250")})
    np.savez(out, incumbent=np.stack(res["incumbent"]), cap250=np.stack(res["cap250"]), scheme=np.array(res["scheme"]),
             people=np.array(sorted(set(d["subject"]))))


if __name__ == "__main__":
    main(*sys.argv[1:])
