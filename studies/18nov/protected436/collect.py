"""Validate complete prediction matrices before loading labels and using the official metric."""
import json
import sys
from pathlib import Path

import numpy as np

from returning_assets import ROOT, sha

REPO = ROOT / "neural-interfaces-2026"
sys.path[:0] = [str(REPO / "experiments"), str(REPO / "2026-competition/tracks/bci_decoding")]
from protocol import load, score


def returning():
    old = ROOT / "ni26_js321_20261117_returning_pair_cpu1"
    banks = []
    proofs = []
    for i in range(16):
        folder = Path(f"cell_{i}")
        proof = json.loads((folder / "proof.json").read_text())
        run = folder / "runtime.npz"
        if proof["index"] != i or proof["source"] != sha("source.sha256") or proof["runtime"] != sha(run):
            raise ValueError("returning provenance/coverage differs")
        with np.load(run) as a, np.load(old / folder / "predictions.npz") as b:
            if not np.array_equal(a["index"], b["index"]) or not np.isin(a["labels"], [0, 1, 2]).all():
                raise ValueError("returning control rows or classes differ")
            banks.append((a["index"], a["labels"], b["base_labels"], b["js_labels"]))
        proofs.append(proof)
    d = load()
    scores = np.empty((2, 8, 3))
    for i, (index, *labels) in enumerate(banks):
        person = ("P04", "P10")[i // 8]
        mask = (d["subject"] == "Dreyer2026Proteus/" + person) & (d["session"] != "1")
        if not np.array_equal(np.flatnonzero(mask), index):
            raise ValueError("official returning rows differ")
        for k, values in enumerate(labels):
            m = score(d["y"][index], values, d, mask)
            if m["n_cells"] != 4:
                raise ValueError("expected four session/interface cells")
            scores[i // 8, i % 8, k] = 100*m["balanced_accuracy_cell_mean"]
    delta = scores[:, :, 0] - scores[:, :, 1]
    w = np.array([.2, .8])
    rng = np.random.default_rng(436)
    p = rng.integers(2, size=(10000, 2))
    o = rng.integers(8, size=(10000, 2, 8))
    boot = (delta[p[:, :, None], o].mean(2)*w[p]).sum(1)/w[p].sum(1)
    return dict(people=["P04","P10"], arms=["protected","base","JS"], percent=scores.mean(1).tolist(),
                weighted_delta_pp=float(w @ delta.mean(1)), descriptive_ci95=np.quantile(boot,[.025,.975]).tolist(),
                person_delta_pp=delta.mean(1).tolist(), proofs=proofs)


def external():
    from data import permutation
    old = ROOT / "ni26_qualify385_v1_20261118"
    data = REPO / "data/scherer2015/windows.npz"
    if sha(data) != "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91":
        raise ValueError("external data identity differs")
    with np.load(data) as z:
        people, days, interface = (z[k] for k in ("subject","session","interface"))
    stream = np.char.add(np.char.add(people, "|"), days)
    orders = [permutation(stream,None,"full",s) for s in (0,1)]
    pred = np.full((2,2,len(people)), -1)
    active = {}
    for i, person in enumerate(sorted(set(people))):
        for arm, path in enumerate((old / f"cap_{i}.json", Path(f"policy_{i}.json"))):
            bank = json.loads(path.read_text())
            if len(bank) != 4:
                raise ValueError("expected two orders and both sessions")
            for key, rec in bank.items():
                p, day, s = key.split("|")
                s = int(s)
                rows = orders[s][(people == p) & (days == day)]
                if p != person or not np.array_equal(rows,rec["rows"]) or np.any(pred[arm,s,rows] != -1):
                    raise ValueError("external control rows or order differ")
                pred[arm,s,rows] = rec["labels"]
                if arm == 1:
                    active[key] = rec["activity"]["recognized"]
    if not np.isin(pred, [0,1,2]).all() or len(active) != 36:
        raise ValueError("incomplete external predictions")
    with np.load(data) as z:
        y = z["y"]
    cells = np.empty((9,2,2,2))
    for i,p in enumerate(sorted(set(people))):
        for j,day in enumerate(sorted(set(days[people == p]))):
            m = (people==p)&(days==day)
            if not np.array_equal(np.unique(y[m]), [0,1,2]):
                raise ValueError("missing class")
            cells[i,j] = np.mean([(pred[:,:,m & (y==c)]==c).mean(2) for c in range(3)],axis=0).T*100
    for a in (0,1):
        for s in (0,1):
            m = score(y,pred[a,s],dict(stream=stream,interface=interface),np.ones(len(y),bool))
            if m["n_cells"] != 18 or not np.isclose(100*m["balanced_accuracy_cell_mean"],cells[:,:,s,a].mean()):
                raise ValueError("official external cell scores differ")
    diff = cells[:,:,:,1] - cells[:,:,:,0]
    rng = np.random.default_rng(436)
    p = rng.integers(9,size=(10000,9,1,1))
    d = rng.integers(2,size=(10000,9,2,1))
    o = rng.integers(2,size=(10000,9,2,2))
    return dict(arms=["cap250","protected"], percent=cells.mean((0,1,2)).tolist(),
                delta_pp=float(diff.mean()), ci80=np.quantile(diff[p,d,o].mean((1,2,3)),[.1,.9]).tolist(),
                order_delta_pp=diff.mean((0,1)).tolist(), person_delta_pp=diff.mean((1,2)).tolist(),
                changed=int((pred[0]!=pred[1]).sum()), recognized=active,
                people=sorted(set(people)), source=sha("source.sha256"))


if __name__ == "__main__":
    if "OK" not in Path("replay_check.txt").read_text().splitlines():
        raise ValueError("expected verified replay")
    result = {"external":external,"returning":returning}[sys.argv[1]]()
    Path("result436.json").write_text(json.dumps(result, indent=2)+"\n")
