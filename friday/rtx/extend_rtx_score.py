"""Compare the unchanged packages' exact A100 and RTX decisions on Scherer."""

import json
from pathlib import Path

import numpy as np

from data import permutation
from frozen_rtx_score import ARMS, DATA, PRIOR, ROOT, sha
from protocol import score


PREVIOUS = ROOT / "ni26_friday_frozen_rtx_20261118"
EXPOSED = (0, 5)
NEW = (1, 2, 3, 4, 6, 7, 8)
HARDWARE = ("A100", "RTX2080Ti")


def aligned_bank(people, days, sessions, cohort):
    orders = [permutation(sessions, None, "full", s) for s in (0, 1)]
    pred = np.full((2, 2, 2, len(people)), -1, np.int8)
    changes, receipts, times = {}, {}, {}
    for i, who in enumerate(cohort):
        changes[who], times[who] = {}, {}
        current = PREVIOUS if i in EXPOSED else Path(".")
        for ai, arm in enumerate(ARMS):
            rtx = current / f"rtx_{arm}_{i}.json"
            a100 = PRIOR / f"{arm}_{i}.json"
            receipts[rtx.name], receipts[a100.name] = sha(rtx), sha(a100)
            actual, baseline = (json.loads(p.read_text()) for p in (rtx, a100))
            assert actual["person"] == i and actual["arm"] == arm
            assert actual["torch"] == "2.14.1+cu130" and "RTX 2080 Ti" in actual["device"]
            assert set(actual["sessions"]) == set(baseline) and len(baseline) == 4
            runtime, moved = [], []
            for oi, order in enumerate(orders):
                for day in sorted(set(days[people == who])):
                    key = f"{who}|{day}|{oi}"
                    expected, row = baseline[key], actual["sessions"][key]
                    rows = order[(people == who) & (days == day)]
                    assert rows.tolist() == expected["rows"] == row["rows"]
                    assert expected["activity"] == row["activity"]
                    assert (pred[:, ai, oi, rows] == -1).all()
                    pred[0, ai, oi, rows] = expected["labels"]
                    pred[1, ai, oi, rows] = row["labels"]
                    change = np.flatnonzero(np.asarray(expected["labels"]) != row["labels"])
                    moved.extend(dict(key=key, age=int(age), row=int(rows[age])) for age in change)
                    runtime.append(row["seconds"])
            changes[who][arm] = moved
            times[who][arm] = dict(load_seconds=actual["load_seconds"], cold=runtime[0],
                warm=runtime[1:], peak_cuda_reserved_bytes=actual["peak_cuda_reserved_bytes"])
    assert np.isin(pred, (0, 1, 2)).all()
    bank = Path("rtx_nine_predictions.npz")
    np.savez_compressed(bank, predictions=pred)
    proof = Path("rtx_nine_before_labels.json")
    proof.write_text(json.dumps(dict(inputs=receipts, bank=sha(bank), scorer=sha(Path("score.sha256")),
                                     source=sha(Path("extend_source.sha256")),
                                     original_pair=sha(Path("rtx_first_pair.sha256"))), indent=2) + "\n")
    return pred, changes, times, proof


def intervals(delta):
    n = len(delta)
    rng = np.random.default_rng(459)
    p = rng.integers(n, size=(10000, n, 1, 1))
    d = rng.integers(2, size=(10000, n, 2, 1))
    o = rng.integers(2, size=(10000, n, 2, 2))
    return np.quantile(delta[p, d, o].mean((1, 2, 3)), (.1, .9)).tolist()


def results(pred, labels, people, days, meta, cohort, indexes):
    chosen = [cohort[i] for i in indexes]
    mask = np.isin(people, chosen)
    stats = np.empty((2, 2, len(chosen), 2, 2))
    for pi, who in enumerate(chosen):
        for di, day in enumerate(sorted(set(days[people == who]))):
            rows = [(people == who) & (days == day) & (labels == c) for c in range(3)]
            assert all(r.any() for r in rows)
            stats[:, :, pi, di, :] = 100 * np.mean([
                (pred[:, :, :, row] == c).mean(-1) for c, row in enumerate(rows)
            ], axis=0)
    for hi in range(len(HARDWARE)):
        for ai in range(len(ARMS)):
            for oi in range(2):
                result = score(labels[mask], pred[hi, ai, oi, mask], meta, mask)
                assert result["n_cells"] == 2 * len(chosen)
                np.testing.assert_allclose(stats[hi, ai, :, :, oi].mean(),
                    100 * result["balanced_accuracy_cell_mean"], atol=1e-10, rtol=0)
    diff = stats[:, 1] - stats[:, 0]
    return dict(people=chosen, outcomes={hardware: dict(
        absolute=dict(zip(ARMS, stats[hi].mean((1, 2, 3)).tolist())),
        pp=float(diff[hi].mean()), paired80=intervals(diff[hi]),
        people=dict(zip(chosen, diff[hi].mean((1, 2)).tolist())),
        orders=diff[hi].mean((0, 1)).tolist()) for hi, hardware in enumerate(HARDWARE)},
        drift_pp=float((diff[1] - diff[0]).mean()), drift_paired80=intervals(diff[1] - diff[0]),
        label_drift={arm: np.count_nonzero(pred[0, ai][:, mask] != pred[1, ai][:, mask], axis=1).tolist()
                     for ai, arm in enumerate(ARMS)})


def main():
    assert Path("replay.log").read_text().splitlines()[-1] == "OK"
    assert sha(DATA) == "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"
    with np.load(DATA, allow_pickle=False) as source:
        people, days, interfaces = (source[key] for key in ("subject", "session", "interface"))
    cohort = sorted(set(people))
    assert len(cohort) == 9 and len(people) == 2130
    sessions = np.char.add(np.char.add(people, "|"), days)
    pred, changes, times, proof = aligned_bank(people, days, sessions, cohort)
    with np.load(DATA, allow_pickle=False) as source:
        labels = source["y"]
    meta = dict(stream=sessions, interface=interfaces)
    report = {name: results(pred, labels, people, days, meta, cohort, ids)
              for name, ids in (("remaining_seven", NEW), ("pooled_nine", range(9)))}
    qualification = json.loads((PRIOR / "result.json").read_text())
    exact = list(report["pooled_nine"]["outcomes"]["A100"]["absolute"].values())
    np.testing.assert_allclose(exact, qualification["acc"], atol=1e-10, rtol=0)
    for who, timing in times.items():
        timing["warm_cap_old_ratio"] = sum(timing["cap"]["warm"]) / sum(timing["old"]["warm"])
    result = dict(cohorts=report, timing=times, label_changes=changes, proof=sha(proof))
    Path("rtx_nine_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
