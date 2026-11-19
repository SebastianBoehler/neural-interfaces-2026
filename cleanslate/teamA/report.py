"""Compare Team A to the archived cap250 predictions with nested person/order resampling."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import data as harness  # noqa: E402


def table(path, mode):
    with np.load(path, allow_pickle=False) as z:
        cells = [{k: z[k][i].item() for k in z.files} for i in np.flatnonzero(z["mode"] == mode)]
    return cells


def cap250_new(path, d):
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["index"], np.arange(len(d["y"]))):
            raise ValueError("Archived cap250 row index differs from PROTEUS")
        pred = z["stock|terminal"].argmax(-1)
    return harness.cell_table(d, pred, np.ones(len(pred), bool), 0, "new")


def cap250_returning(root, d):
    cells = []
    for person in harness.RETURNING:
        for session in ("2", "3"):
            folders = sorted(root.glob(f"*_{person}_{session}"))
            if len(folders) != 8:
                raise ValueError(f"Expected eight archived orders for {person}/{session}, got {len(folders)}")
            for order, folder in enumerate(folders):
                with np.load(folder / f"{folder.name}.npz", allow_pickle=False) as z:
                    rows, labels = z["rows"], z["stock|labels"]
                expected = (d["person"] == person) & (d["session"] == session)
                np.testing.assert_array_equal(np.sort(rows), np.flatnonzero(expected))
                pred = np.full(len(d["y"]), -1, dtype=np.int8)
                pred[rows] = labels
                cells.extend(harness.cell_table(d, pred, expected, order, "ret"))
    return cells


def by_order(cells, people):
    orders = sorted({c["order"] for c in cells})
    return np.asarray([[np.mean([c["ba"] for c in cells if c["order"] == o and c["person"] == p])
                        for o in orders] for p in people])


def interval(draws):
    return np.percentile(draws, [10, 90]).round(3).tolist()


def compare(new, old, rng):
    people = sorted({c["person"] for c in new})
    if {c["person"] for c in old} != set(people):
        raise ValueError("Candidate and reference people differ")
    a, b = by_order(new, people), by_order(old, people)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Missing person/order combination")
    n = 10000
    p = rng.integers(len(people), size=(n, len(people), 1))

    def resample(x, person):
        order = rng.integers(x.shape[1], size=(*person.shape[:-1], x.shape[1]))
        return x[person, order].mean(axis=-1)

    da = resample(a, p) - resample(b, p)
    boot = da.mean(axis=-1)
    result = dict(
        candidate_cell_ba=float(np.mean([c["ba"] for c in new])),
        cap250_cell_ba=float(np.mean([c["ba"] for c in old])),
        person_equal_delta_pp=float((a.mean(1) - b.mean(1)).mean()),
        person_nested_80_pp=interval(boot),
        by_person_pp=dict(zip(people, (a.mean(1) - b.mean(1)).round(3))),
        by_person_ba=dict(zip(people, a.mean(1).round(3))),
        orders_candidate=a.shape[1], orders_cap250=b.shape[1],
    )
    return result, boot, da


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=Path, required=True)
    p.add_argument("--cap250-new", type=Path, required=True)
    p.add_argument("--cap250-ret", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = harness.load(args.data)
    rng = np.random.default_rng(20261120)
    base_new = cap250_new(args.cap250_new, data)
    base_ret = cap250_returning(args.cap250_ret, data)
    new, nb, _ = compare(table(args.cells, "new"), base_new, rng)
    ret, rb, _ = compare(table(args.cells, "ret"), base_ret, rng)
    weighted = 0.57 * nb + 0.43 * rb
    result = dict(
        new=new, returning=ret,
        sealed_proxy=dict(
            assumption="Unverified mix: 57% new, 43% returning. Only P04 and P10 represent returning.",
            equal_people_delta_pp=round(float(weighted.mean()), 3), equal_people_i80_pp=interval(weighted),
            p04_20_p10_80_delta_pp=round(0.57 * new["person_equal_delta_pp"] + 0.43 * (
                0.2 * ret["by_person_pp"]["P04"] + 0.8 * ret["by_person_pp"]["P10"]), 3),
        ),
        reference_limit="Paired by person, delivery orders differ. Reference uses legacy full0 and Lane A tapes.",
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
