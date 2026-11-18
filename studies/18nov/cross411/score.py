"""Within-window early/late contrasts on the two frozen Scherer deliveries."""
import argparse
import json
from pathlib import Path

import numpy as np
from data import permutation


def evaluate(bank, dataset, smoke, seeds):
    with np.load(dataset) as archive:
        people, days, labels = (archive[name] for name in ('subject', 'session', 'y'))
    streams = np.char.add(np.char.add(people, '|'), days)
    orders = [permutation(streams, None, 'full', seed) for seed in seeds]
    primary = json.loads((bank / 'result.json').read_text())
    full_cells, person_cells = [], []
    count = missing = 0
    per_person = {}
    for index, person in enumerate(sorted(set(people))):
        banks = [json.loads((bank / f'{arm}_{index}.json').read_text()) for arm in ('old', 'cap')]
        day_cells = []
        for day in sorted(set(days[people == person])):
            rows = np.flatnonzero((people == person) & (days == day))
            positions = np.empty((2, len(rows)), int)
            correct = np.empty((2, 2, len(rows)), int)
            for seed, order in enumerate(orders):
                pinned = order[np.isin(order, rows)]
                indices = np.searchsorted(rows, pinned)
                positions[seed, indices] = np.arange(1, len(rows) + 1)
                for arm, records in enumerate(banks):
                    rec = records[f'{person}|{day}|{seeds[seed]}']
                    assert np.array_equal(pinned, rec['rows'])
                    assert np.isin(rec['labels'], [0, 1, 2]).all()
                    correct[arm, seed, indices] = rec['labels'] == labels[pinned]
            delta = correct[1] - correct[0]
            early0 = (positions[0] <= 50) & (positions[1] >= 91)
            early1 = (positions[1] <= 50) & (positions[0] >= 91)
            crossed = early0 | early1
            count += int(crossed.sum())
            contrasts = np.where(early0, delta[1] - delta[0], delta[0] - delta[1])
            class_cells = []
            for label in range(3):
                class_mask = labels[rows] == label
                full_cells.append(correct[:, :, class_mask].mean(axis=(1, 2)))
                mask = crossed & class_mask
                if mask.any():
                    class_cells.append(100 * contrasts[mask].mean())
                else:
                    missing += 1
            assert class_cells
            day_cells.append(np.mean(class_cells))
        person_cells.append(np.mean(day_cells))
        per_person[str(person)] = float(person_cells[-1])
        if smoke:
            return {'smoke_person': str(person), 'matched_rows': count}
    original = 100 * np.mean(full_cells, axis=0)
    np.testing.assert_allclose(original, primary['acc'], atol=1e-10, rtol=0)
    draws = np.random.default_rng(411).choice(person_cells, size=(10000, len(person_cells)))
    return dict(matched_rows=count, total_rows=len(labels), missing_cells=missing,
                package_acc=original.tolist(), paired_late_minus_early_pp=np.mean(person_cells),
                person_bootstrap80=np.quantile(draws.mean(axis=1), [.1, .9]).tolist(),
                person_pp=per_person)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replica', action='store_true')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.bank, args.dataset, args.smoke, (411, 1411) if args.replica else (0, 1)), indent=2))
