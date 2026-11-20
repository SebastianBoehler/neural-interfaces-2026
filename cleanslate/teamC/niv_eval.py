"""NI-V new-person score of a team C package on Scherer 2015 (package must not have trained on Scherer).

python cleanslate/teamC/niv_eval.py --pkg DIR --orders 71000,71001 --variants A1,A2 --out FILE.json
Each visit of each Scherer person is streamed on deepcopy(model) + reset_state(), one window per predict, in the
niv/orders.py "full" permutation of that seed. Cell = (person, visit) balanced accuracy; score = cell mean.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
from niv.orders import permutation  # noqa: E402
from stream_eval import VARIANTS, load_model  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--orders", default="71000,71001")
    ap.add_argument("--variants", default="A1,A2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/scherer2015/windows.npz"))
    a = ap.parse_args()
    tj = json.load(open(Path(a.pkg) / "train.json"))
    assert not any(s.lower().startswith("scherer") for s in tj["streams"]), "package trained on Scherer"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data)
    stream = np.char.add(np.char.add(d["subject"].astype(str), "|"), d["session"].astype(str))
    res = []
    for seed in map(int, a.orders.split(",")):
        p = permutation(stream, d["run"], "full", seed)
        for v in a.variants.split(","):
            model = load_model(a.pkg, dev, VARIANTS[v])
            cells = {}
            for st in dict.fromkeys(stream):
                rows = p[stream[p] == st]
                m = copy.deepcopy(model)
                m.reset_state()
                lab = [int(m.predict(torch.as_tensor(d["X"][r:r + 1]))[0]) for r in rows]
                cells[st] = 100 * balanced_accuracy_score(d["y"][rows], lab)
            res.append(dict(order=seed, variant=v, ba=float(np.mean(list(cells.values()))), cells=cells))
            print("NIV", seed, v, round(res[-1]["ba"], 2), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
