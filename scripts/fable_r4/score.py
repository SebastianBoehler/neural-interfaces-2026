"""Validate complete fold tapes and score paired official cell weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from fold import DATA_SHA, D, LEAD_SHA, ORACLE, digest, streams
from personal import CHECKPOINTS


def cell_weights(y, interfaces):
    weights = np.empty(len(y))
    for interface in np.unique(interfaces):
        cell = interfaces == interface
        classes = np.unique(y[cell])
        for cls in classes:
            selected = cell & (y == cls)
            weights[selected] = 1 / (len(classes) * selected.sum())
    return weights


def stats(bank, people):
    def measure(values):
        lead, candidate, mass = np.moveaxis(values, -1, 0)
        return 100 * (candidate - lead) / mass

    point = bank.sum(axis=(0, 1))
    rng = np.random.default_rng(20261121)
    oi = rng.integers(len(bank), size=(20_000, len(bank), 1))
    pi = rng.integers(len(people), size=(20_000, 1, len(people)))
    sampled = measure(bank[oi, pi].sum(axis=(1, 2)))
    return dict(lead=float(100 * point[0] / point[2]), candidate=float(100 * point[1] / point[2]),
                mean=float(measure(point)), ci80=np.quantile(sampled, [.1, .9]).tolist(),
                people=dict(zip(people, measure(bank.sum(axis=0)).tolist(), strict=True)),
                by_order=measure(bank.sum(axis=1)).tolist())


def read_tapes(data, root, orders, modes):
    reports, provenance = {}, {}
    for mode, people in modes:
        arrays = {name: np.zeros((len(orders), len(people), 3)) for name in ("whole", "fixed_after100", "ungated_after100", "1-40",
                                                                          "41-70", "71-100", "101-140", "141+")}
        fit_times = []
        accepted = np.zeros(4, dtype=int)
        attempted = np.zeros(4, dtype=int)
        window_count = changed = active = recognized = 0
        for p, person in enumerate(people):
            fold = f"{mode}_{person}"
            keys = ("fold", "source_sha", "lead_sha", "bank_sha", "bank_source_sha", "weights_sha", "data_sha")
            for first in range(0, len(orders), 2):
                group = orders[first:first + 2]
                path = root / f"{fold}_{group[0]}.npz"
                with np.load(path, allow_pickle=False) as z:
                    prov = {k: str(z[k]) for k in keys}
                    if (prov["fold"] != fold or tuple(z["orders"]) != group or prov["data_sha"] != DATA_SHA
                            or prov["lead_sha"] != LEAD_SHA or provenance.setdefault(fold, prov) != prov
                            or any(len(prov[k]) != 64 for k in keys[1:])):
                        raise ValueError(f"inconsistent tape identity in {path}")
                    for o, order in enumerate(group, first):
                        for stream, rows in D.plans(data, order, streams(data, fold)).items():
                            key = f"{order}_{stream.split('|')[1]}"
                            np.testing.assert_array_equal(rows, z[f"rows_{key}"])
                            expected = data["y"][rows]
                            columns = [z[f"{name}_{key}"] for name in ("lead", "enrolled", "after100", "ungated100")]
                            if not all(c.shape == rows.shape and np.isin(c, (0, 1, 2)).all() for c in columns):
                                raise ValueError(f"invalid tape labels in {path}/{key}")
                            lead, candidate, fixed, ungated = columns
                            np.testing.assert_array_equal(lead[:40], candidate[:40])
                            np.testing.assert_array_equal(lead[:100], fixed[:100])
                            np.testing.assert_array_equal(lead[:100], ungated[:100])
                            weights = cell_weights(expected, data["interface"][rows])
                            errors = {name: z[f"{name}_{key}"] for name in ("loo", "selected", "fit_seconds")}
                            if any(v.shape != (len(CHECKPOINTS),) for v in errors.values()):
                                raise ValueError(f"invalid checkpoint receipts in {path}/{key}")
                            fits = errors["fit_seconds"]
                            fit_times.extend(fits[np.isfinite(fits)])
                            attempted += np.isfinite(fits)
                            accepted += errors["loo"] >= .75
                            rec = int(z[f"recognition_{key}"])
                            if rec:
                                np.testing.assert_array_equal(lead[rec - 1:], candidate[rec - 1:])
                            recognized += bool(rec)
                            active += len(z[f"active_{key}"])
                            window_count += len(rows)
                            changed += np.count_nonzero(lead != candidate)
                            bounds = ((0, None), (100, None), (100, None), (0, 40), (40, 70), (70, 100), (100, 140), (140, None))
                            answers = {"fixed_after100": fixed, "ungated_after100": ungated}
                            for (name, bank), (lo, hi) in zip(arrays.items(), bounds, strict=True):
                                hit = answers.get(name, candidate) == expected
                                s = slice(lo, hi)
                                bank[o, p] += [weights[s] @ (lead[s] == expected[s]), weights[s] @ hit[s], weights[s].sum()]
        if not np.allclose(arrays["whole"][:, :, 2].sum(1), 28 if mode == "new" else 8, rtol=0, atol=1e-10):
            raise ValueError(f"incorrect number of official cells for {mode}")
        reports[mode] = {name: stats(bank, people) for name, bank in arrays.items()}
        reports[mode]["activity"] = dict(attempted=attempted.tolist(), accepted=accepted.tolist(),
                                        fit_max_sec=float(max(fit_times, default=0.)),
                                        windows=window_count, changed=changed, active=active,
                                        recognized_streams=recognized)
    return reports, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("oracle",))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    if digest(a.data) != DATA_SHA:
        raise ValueError("PROTEUS data SHA-256 differs")
    orders, modes = ORACLE, (("new", D.PEOPLE),)
    scores, provenance = read_tapes(D.load(a.data), a.run / a.stage, orders, modes)
    gain = scores["new"]["fixed_after100"]["mean"]
    result = dict(stage=a.stage, orders=orders, scores=scores, provenance=provenance, release_pseudo=gain >= 4.)
    out = a.run / f"{a.stage}.json"
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(scores=scores, receipt_sha=digest(out), release_pseudo=result["release_pseudo"])), flush=True)


if __name__ == "__main__":
    main()
