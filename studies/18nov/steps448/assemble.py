"""Restore the common scorer's paired-bank layout and retain warm repeat changes."""
import json
from pathlib import Path

summary = {}
for i in range(9):
    out = {}
    for arm in ('stock', 'lower', 'zero', 'alternate'):
        records = json.loads(Path(f'{arm}_{i}.json').read_text())
        assert len(records) == 4
        for key, rec in records.items():
            rows = rec.pop('rows')
            target = out.setdefault(key, dict(rows=rows, arms={}))
            assert rows == target['rows']
            if 'warm_differences' in rec:
                summary[f'{key}|{arm}'] = rec['warm_differences']
            target['arms'][arm] = rec
    assert len(out) == 4
    Path(f'person_{i}.json').write_text(json.dumps(out))
assert len(summary) == 36
Path('warm_differences.json').write_text(json.dumps(summary))
