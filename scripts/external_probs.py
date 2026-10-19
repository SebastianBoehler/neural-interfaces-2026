"""Score the round-3 ensemble with members trained on added external data.

Usage: python scripts/external_probs.py OUT.json PROTOCOLS MEMBER=NAME [...]
  e.g. ... dev_loso,dev_session shallow_raw=shallow_raw_ft@scherer2015~n
Members are v2, shallow_ea, shallow_raw and fbcnet_ea (deep members averaged
over their saved seeds), mixed with equal quarters as in ensemble_all_r3 and
causally rebalanced (alpha 0.5, k 5). Per protocol it scores the PROTEUS-only
reference, each swap alone and all swaps together. Probabilities:
runs/experiments/probs{,_cloud} (reference) and runs/experiments/probs_r5
(external runs; seed-averaged as available, usually seeds 0-2).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import protocol  # noqa: E402
from rebalance_probs import rebalance  # noqa: E402

MEMBERS = ("v2", "shallow_ea", "shallow_raw", "fbcnet_ea")
REF = ("probs", "probs_cloud")
EXT = ("probs_r5",)


def member(name, proto, dirs):
    runs = {}
    for d in dirs:
        for f in sorted((ROOT / "runs/experiments" / d).glob(f"{name}-{proto}-s*.npz")):
            runs.setdefault(f.stem.rsplit("-s", 1)[1], f)  # first directory wins
    if not runs:
        return None
    loaded = [np.load(f) for f in runs.values()]
    idx = loaded[0]["index"]
    assert all((z["index"] == idx).all() for z in loaded)
    return np.mean([z["proba"] for z in loaded], 0), idx, len(loaded)


def main():
    out_path, protos = sys.argv[1], sys.argv[2].split(",")
    swaps = dict(a.split("=", 1) for a in sys.argv[3:])
    assert set(swaps) <= set(MEMBERS), swaps
    d = protocol.load()
    res = {"swaps": swaps}
    for proto in protos:
        ref = {m: member(m, proto, REF) for m in MEMBERS}
        ext = {m: member(n, proto, EXT) for m, n in swaps.items()}
        ext = {m: e for m, e in ext.items() if e}
        idx = ref["v2"][1]
        assert all((r[1] == idx).all() for r in list(ref.values()) + list(ext.values()))
        mask = np.zeros(len(d["y"]), bool)
        mask[idx] = True
        y, streams = d["y"][idx], d["stream"][idx]

        def cell(P, alpha=0.5):
            Q = rebalance(P, streams, alpha, 5.0) if alpha else P
            return round(protocol.score(y, Q.argmax(1), d, mask)
                         ["balanced_accuracy_cell_mean"], 4)

        r = {}
        for m, e in ext.items():
            r[f"{m} (ref, {ref[m][2]} seeds)"] = cell(ref[m][0], 0)
            r[f"{swaps[m]} ({e[2]} seeds)"] = cell(e[0], 0)
        variants = {"reference": set()} | {f"swap {m}": {m} for m in ext}
        if len(ext) > 1:
            variants["swap all"] = set(ext)
        for label, sw in variants.items():
            P = np.mean([(ext if m in sw else ref)[m][0] for m in MEMBERS], 0)
            r[f"ens {label}"] = cell(P, 0)
            r[f"ens_rb {label}"] = cell(P)
        res[proto] = r
        print(proto, json.dumps(r, indent=1))
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
