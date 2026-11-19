"""Verify blind predictions and report the paired external F5 comparison."""

import json
from pathlib import Path

import numpy as np

from data import permutation
from external_inputs import ARMS, MODELS, ORDERS, SEEDS, digest, verify_models
from protocol import score


PRIOR = Path("../ni26_qualify385_v1_20261118")
DATA = Path("../neural-interfaces-2026/data/scherer2015/windows.npz")
DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"
STOCK = 48.45568783068782


def resamples():
    rng = np.random.default_rng(459)
    return (
        rng.integers(9, size=(10_000, 9, 1, 1)),
        rng.integers(2, size=(10_000, 9, 2, 1)),
        rng.integers(2, size=(10_000, 9, 2, 2)),
    )


def summarize(delta, people, draws):
    return dict(pp=float(delta.mean()),
                paired80=np.quantile(delta[draws].mean((1, 2, 3)), [.1, .9]).tolist(),
                people=dict(zip(people, delta.mean((1, 2)).tolist())),
                orders=dict(zip(ORDERS, delta.mean((0, 1)).tolist())))


def contrast(left, right, cells, labels, predicted, people, draws):
    a, b = ARMS.index(left), ARMS.index(right)
    hits = predicted[a] == labels
    prev = predicted[b] == labels
    return dict(**summarize(cells[a] - cells[b], people, draws),
                changed=int(np.count_nonzero(predicted[a] != predicted[b])),
                corrections=int(np.count_nonzero(hits & ~prev)),
                regressions=int(np.count_nonzero(prev & ~hits)))


def blind_bank(subjects, days, streams, people, visits):
    expected_orders = [permutation(streams, None, "full", order) for order in ORDERS]
    shape = (len(ARMS), len(ORDERS), len(subjects))
    pred = np.full(shape, -1, dtype=np.int8)
    model_hashes = verify_models(Path("weights"))
    receipt = dict(actors={}, source=digest("external_source.sha256"),
                   scorer=digest(Path(__file__)), models=model_hashes)
    timing = {name: [] for name in ("stock", *MODELS)}
    for i, who in enumerate(people):
        path = Path(f"external_{i}.json")
        receipt["actors"][path.name] = digest(path)
        result = json.loads(path.read_text())
        archive = PRIOR / f"cap_{i}.json"
        assert (result["archive"], result["source"]) == (digest(archive), receipt["source"])
        assert result["models"] == model_hashes and "A100" in result["gpu"]
        prior = json.loads(archive.read_text())
        assert set(result["sessions"]) == set(prior) and len(prior) == 4
        for oi, order in enumerate(ORDERS):
            for day in visits[who]:
                key = f"{who}|{day}|{order}"
                record, control = result["sessions"][key], prior[key]
                rows = expected_orders[oi][(subjects == who) & (days == day)]
                assert record["rows"] == rows.tolist() == control["rows"]
                assert record["counters"] == control["activity"]
                assert set(record["arms"]) == set(ARMS) and record["arms"]["stock"] == control["labels"]
                for ai, arm in enumerate(ARMS):
                    values = record["arms"][arm]
                    assert (pred[ai, oi, rows] == -1).all()
                    assert len(values) == len(rows) and np.isin(values, (0, 1, 2)).all()
                    pred[ai, oi, rows] = values
                assert set(record["runtime"]) == set(timing)
                for name, values in record["runtime"].items():
                    assert all(np.isfinite(v) and v > 0 for v in values.values())
                    timing[name].append(values)
    assert np.isin(pred, (0, 1, 2)).all()
    bank = Path("external_predictions.npz")
    np.savez_compressed(bank, arms=ARMS, predictions=pred)
    receipt["prediction_bank"] = digest(bank)
    Path("external_before_labels.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return pred, timing


def cell_scores(pred, labels, subjects, days, streams, interfaces, people, visits):
    cells = np.empty((len(ARMS), 9, 2, len(ORDERS)))
    for i, who in enumerate(people):
        for j, day in enumerate(visits[who]):
            masks = [(subjects == who) & (days == day) & (labels == c) for c in range(3)]
            assert all(mask.any() for mask in masks)
            recalls = [(pred[:, :, mask] == c).mean(2) for c, mask in enumerate(masks)]
            cells[:, i, j, :] = 100 * np.mean(recalls, axis=0)
    for arm in range(len(ARMS)):
        for order in range(len(ORDERS)):
            result = score(labels, pred[arm, order], dict(stream=streams, interface=interfaces),
                           np.ones(len(labels), bool))
            assert result["n_cells"] == 18
            np.testing.assert_allclose(cells[arm, :, :, order].mean(),
                                       100 * result["balanced_accuracy_cell_mean"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(cells[0].mean(), STOCK, atol=1e-10, rtol=0)
    return cells


def main():
    assert Path("replay.log").read_text().splitlines()[-1] == "OK"
    assert digest(DATA) == DATA_SHA
    with np.load(DATA, allow_pickle=False) as source:
        subjects, days, interfaces = (source[k] for k in ("subject", "session", "interface"))
    streams = np.char.add(np.char.add(subjects, "|"), days)
    people = sorted(set(subjects))
    visits = {who: sorted(set(days[subjects == who])) for who in people}
    assert len(people) == 9 and len(subjects) == 2130 and all(len(v) == 2 for v in visits.values())
    pred, timing = blind_bank(subjects, days, streams, people, visits)
    with np.load(DATA, allow_pickle=False) as source:
        labels = source["y"]
    cells = cell_scores(pred, labels, subjects, days, streams, interfaces, people, visits)
    draws = resamples()
    contrasts = {}
    for seed in SEEDS:
        true, swap, ordinary = (f"{k}_{seed}" for k in ("true", "wrongswap", "ordinary"))
        genuine = f"{true}_genuine"
        for control in ("stock", f"{swap}_genuine", f"{ordinary}_genuine", f"{true}_blank"):
            contrasts[f"{genuine}-{control}"] = contrast(genuine, control, cells, labels, pred, people, draws)
        for other in (swap, ordinary):
            own_gain = cells[ARMS.index(genuine)] - cells[ARMS.index(f"{true}_blank")]
            control_gain = cells[ARMS.index(f"{other}_genuine")] - cells[ARMS.index(f"{other}_blank")]
            contrasts[f"{true}-{other}_history_interaction"] = summarize(own_gain - control_gain, people, draws)
    runtime = {}
    for name, streams_timing in timing.items():
        runtime[name] = dict(paired_streams=len(streams_timing),
                             total_seconds=sum(v["seconds"] for v in streams_timing),
                             median_stream_warm_p50_ms=float(np.median([v["warm_p50_ms"] for v in streams_timing])),
                             median_stream_warm_p90_ms=float(np.median([v["warm_p90_ms"] for v in streams_timing])))
    result = dict(people=people, orders=ORDERS, decisions_per_arm=pred[0].size,
                  accuracy=dict(zip(ARMS, cells.mean((1, 2, 3)).tolist())),
                  contrasts=contrasts, runtime=runtime, proof=digest("external_before_labels.json"))
    Path("external_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("F5 external paired score sealed after exact stock, canonical order and official-cell checks")


if __name__ == "__main__":
    main()
