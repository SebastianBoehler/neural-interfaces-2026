"""Authenticate all fresh-order cap250 outputs before opening the held labels."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from bank_tools import archive_parity, seal, sha
from metrics import boot_draws, counts, interval, timing
from protocol import score
from settings import ACTIVE, ARMS, SEEDS, schedule


BASE = Path("/mnt/lustre/work/gehler/gfh527")
DATA = BASE / "neural-interfaces-2026/data/scherer2015/windows.npz"
PRIOR = BASE / "ni26_cross411b_20261118"
ROWS = 2130


def read(path):
    return json.loads(path.read_text())


def authenticate(path):
    digest = sha(path)
    assert path.with_suffix(".sha256").read_text().strip() == digest
    return read(path), digest


def verify(source):
    assert source == sha("source.sha256")
    smoke, smoke_digest = authenticate(Path("smoke.json"))
    assert smoke["passed"] and smoke["source"] == source and not smoke["labels_opened"]
    assert smoke["preflight"]["disabled_bitwise"] and smoke["preflight"]["prefix_reset_exact"]
    with np.load(DATA, allow_pickle=False) as f:
        people, days, iface = (f[k] for k in ("subject", "session", "interface"))
    stream = np.char.add(np.char.add(people, "|"), days)
    assert all(len(set(iface[stream == s])) == 1 for s in set(stream))
    roster = sorted(set(people))
    assert roster == read(PRIOR / "result.json")["people"]
    deliveries = schedule(people, days)
    matrix = np.full((len(ARMS), len(SEEDS), ROWS), -1, dtype=np.int8)
    runtime = np.zeros((len(ARMS), len(roster), 2, len(SEEDS)))
    artifacts, stock_anchors, counts_by_arm = {}, {}, {a: Counter() for a in ACTIVE}
    stats = {"max_branch_sep": 0., "max_weight_span": 0., "exposure_error_bound": 0., "moved_rows": 0}
    for i, person in enumerate(roster):
        visits = [f"{person}|{v}" for v in (1, 2)]
        anchor_file = PRIOR / f"cap_{i}.json"
        reference = read(anchor_file)
        stock_anchors[anchor_file.name] = sha(anchor_file)
        path = Path(f"person_{i}.json")
        bank, artifacts[path.name] = authenticate(path)
        assert (bank["who"], bank["index"], bank["source"], bank["smoke_sha"], bank["archive"],
                bank["labels_opened"]) == (person, i, source, smoke_digest, stock_anchors[anchor_file.name], False)
        anchors = bank["archived_controls"]
        assert set(anchors) == {f"{visits[0]}|{seed}" for seed in (411, 1411)}
        for key, value in anchors.items():
            assert value["rows"] == reference[key]["rows"]
            archive_parity(value["run"], reference[key])
        assert set(bank["sessions"]) == {f"{v}|{s}" for v in visits for s in SEEDS}
        for j, seed in enumerate(SEEDS):
            for d, visit in enumerate(visits):
                rows = deliveries[seed][visit]
                record = bank["sessions"][f"{visit}|{seed}"]
                rotation = (i + 2*j + d) % 3
                assert record["order"] == list(ARMS[rotation:]+ARMS[:rotation])
                assert np.array_equal(record["rows"], rows) and (matrix[:, j, rows] == -1).all()
                runs = record["arms"]
                assert set(runs) == set(ARMS) and runs["informative"]["aligned_sha"] != runs["permuted"]["aligned_sha"]
                z, p = (runs[a]["history"] for a in ("informative","permuted"))
                assert z["prior_sha"] == p["prior_sha"] and z["prior_sha"] == smoke["arms"]["informative"]["history"]["prior_sha"]
                assert z["weights"] == p["weights"] and len(z["weights"]) == len(rows)
                w = np.asarray(z["weights"])
                assert np.isfinite(w).all() and ((w >= .09999999) & (w <= .90000001)).all()
                want = np.r_[0., np.cumsum(w)[:-1]]
                err = max(float(np.abs(want - x["branch_mass"]).max()) for x in (z,p))
                assert err <= 2e-12 and record["exposure_error"] <= 2e-12
                assert p["moved"] == 4*(len(rows)//4) and z["moved"] == 0 and p["blocks"] == z["blocks"] == len(rows)//4
                stats["max_branch_sep"] = max(stats["max_branch_sep"], z["separation"])
                stats["max_weight_span"] = max(stats["max_weight_span"], float(np.ptp(w)))
                stats["exposure_error_bound"] = max(stats["exposure_error_bound"], err)
                stats["moved_rows"] += p["moved"]
                for a, arm in enumerate(ARMS):
                    val = runs[arm]
                    act = val["activity"]
                    assert len(val["labels"]) == len(rows) and np.isin(val["labels"], (0,1,2)).all()
                    assert np.isfinite(val["seconds"]) and val["seconds"] > 0
                    assert act["windows"] == len(rows) and act["errors"] == act["ft_over"] == 0
                    assert act["ft_steps"] == 20*act["ft_refits"]
                    assert (val["history"] is None) == (arm == "stock")
                    matrix[a,j,rows] = val["labels"]
                    runtime[a,i,d,j] = val["seconds"]
                    if arm != "stock":
                        base = runs["stock"]["activity"]
                        assert val["history"]["rows"] == len(rows)
                        counts_by_arm[arm].update(k for k in act if act[k] != base[k])
    assert (matrix >= 0).all() and (runtime > 0).all()
    proof = dict(source=source, smoke_sha=smoke_digest, roster=roster, seeds=SEEDS, arms=ARMS,
                 worker_receipts=artifacts, archived_controls=stock_anchors, activity_differences=counts_by_arm,
                 exposure=stats, decisions_sha=hashlib.sha256(matrix.tobytes()).hexdigest(), labels_opened=False)
    return matrix, runtime, proof, people, days, dict(stream=stream, interface=iface)


def cells_for(y, matrix, people, days, meta, roster):
    by_class = np.empty((len(ARMS),len(roster),2,len(SEEDS),3))
    for i, who in enumerate(roster):
        for v, day in enumerate(("1","2")):
            session = (people == who) & (days == day)
            for c in range(3):
                mask = session & (y == c)
                assert mask.any()
                by_class[:,i,v,:,c] = 100*(matrix[:,:,mask] == c).mean(-1)
    cells = by_class.mean(-1)
    for a, arm in enumerate(ARMS):
        for j, seed in enumerate(SEEDS):
            official = score(y,matrix[a,j],meta,np.ones(ROWS,dtype=bool))
            assert official["n_cells"] == 18, (arm,seed)
            np.testing.assert_allclose(np.sort(cells[a,:,:,j].ravel()),
                                       np.sort(100*np.asarray(official["cell_scores"])), atol=1e-10, rtol=0)
            np.testing.assert_allclose(cells[a,:,:,j].mean(),100*official["balanced_accuracy_cell_mean"],atol=1e-10,rtol=0)
    return cells, by_class


def main(source):
    matrix,warm,proof,people,days,meta=verify(source)
    bindings = dict(scoring_source=sha("scoring.sha256"),scoring_inputs=sha("score_inputs.sha256"))
    lock = seal("before_labels.json",{**proof,**bindings})
    with np.load(DATA,allow_pickle=False) as f:
        labels=f["y"]
    assert labels.shape == (ROWS,) and np.isin(labels,(0,1,2)).all()
    cells, by_class = cells_for(labels,matrix,people,days,meta,proof["roster"])
    p,v,o = boot_draws()
    comparisons = (("permuted_vs_stock",1,0),("informative_vs_stock",2,0),("informative_vs_permuted",2,1))
    contrasts = {}
    for name,a,b in comparisons:
        delta = cells[a]-cells[b]
        ci = interval(delta[p,v,o].mean(axis=(1,2,3)))
        orders = dict(zip(SEEDS,delta.mean(axis=(0,1)).tolist(),strict=True))
        outcome = ("supported" if ci[0] > 0 and min(orders.values())>0 else "refuted" if ci[1]<=0 else "undecided")
        contrasts[name] = dict(pp=float(delta.mean()),paired80=ci,order_pp=orders,outcome=outcome,
                               class_pp=(by_class[a]-by_class[b]).mean(axis=(0,1,2)).tolist(),
                               people_pp=dict(zip(proof["roster"],delta.mean(axis=(1,2)).tolist(),strict=True)),
                               decisions=counts(matrix[b],matrix[a],labels))
    rt = {arm:timing(warm[0],warm[a]) for a,arm in enumerate(ACTIVE,start=1)}
    accuracy = all(contrasts[f"informative_vs_{b}"]["outcome"]=="supported" for b in ("stock","permuted"))
    result = dict(scope="FULL-SYSTEM EXTERNAL reused people and fresh delivery", absolute_pct=dict(
        zip(ARMS,cells.mean(axis=(1,2,3)).tolist(),strict=True)), seed_orders=SEEDS,
        forecast_pp=dict(informative_vs_stock=.30,informative_vs_permuted=.22,permuted_vs_stock=.08),
        contrasts=contrasts,a100=rt,accuracy_gate=accuracy,advance=accuracy and rt["informative"]["passes_a100_gate"],
        lock_sha=lock,activity_differences=proof["activity_differences"],exposure=proof["exposure"],
        source=source,**bindings)
    digest = seal("result.json",result)
    print("PASS F-C17 exact paired score",digest,flush=True)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
