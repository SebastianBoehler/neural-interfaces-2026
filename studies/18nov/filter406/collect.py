"""Summarize complete per-session AB/BA timings and equality evidence."""
import json
import sys
from pathlib import Path

import numpy as np

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def collect(folder):
    records = []
    for index in range(16):
        row = json.loads((folder / f'cell_{index}/result.json').read_text())
        if (row['index'], row['fold'], row['order']) != (
            index, ('P04_s23', 'P10_s23')[index // 8], ORDERS[index % 8]
        ):
            raise ValueError('unpaired or missing cell')
        if sorted(r['session'] for r in row['sessions']) != ['2', '3']:
            raise ValueError('both returning sessions required')
        records.append(row)
    sessions = [s for r in records for s in r['sessions']]
    if sum(s['rows'] for s in sessions) != 6096:
        raise ValueError('incomplete prediction coverage')
    for s in sessions:
        values = np.asarray(s['paired_seconds'])
        if values.shape != (2, 2) or not np.isfinite(values).all() or values.min() <= 0:
            raise ValueError('invalid warm AB/BA timing')
    def timings(rows):
        values = np.array([s['paired_seconds'] for s in rows])
        elapsed = values.mean(axis=1)
        return dict(session_orders=len(rows), rows=sum(s['rows'] for s in rows),
                    median_stage_reduction_percent=float(np.median(100*(1-elapsed[:, 1]/elapsed[:, 0]))),
                    total_paired_seconds=values.sum(axis=(0, 1)).tolist(),
                    retained_views=sum(s['pruned_views'] for s in rows))
    sets = {'all': sessions}
    for name, latch in [('latched', True), ('unlatched', False)]:
        subset = [s for s in sessions if (s['latch'] is not None) == latch]
        if subset:
            sets[name] = subset
    return dict(results={k: timings(v) for k, v in sets.items()}, cells=records)


if __name__ == '__main__':
    print(json.dumps(collect(Path(sys.argv[1])), indent=2))
