"""Verify complete rows and repeat decisions before reporting paired timing."""
from pathlib import Path
import json
import sys

runs = [[json.loads(Path(f'{arm}_{rep}.json').read_text())['streams']
         for arm in ('old','cap')] for rep in range(2)]
result = {}
for name, length in [('P10-3',199),('Schirrmeister2017/1',320),('Schirrmeister2017/2',813)]:
    old, cap = runs[0][0][name], runs[0][1][name]
    assert len(set(old['rows'])) == len(old['rows']) == length
    for record in [s[name] for run in runs for s in run]:
        assert record['rows'] == old['rows'] and record['row_hash'] == old['row_hash']
    for arm, original in enumerate((old,cap)):
        other = runs[1][arm][name]
        assert other['labels'] == original['labels'] and other['activity'] == original['activity']
    seconds = [[s[name]['seconds'] for s in run] for run in runs]
    result[name] = dict(
        seconds_old_cap=seconds, ratios=[c/o for o,c in seconds], old_activity=old['activity'],
        cap_activity=cap['activity'], row_sha=cap['row_hash'],
        cap_live_gib=[r[1][name]['allocated']/2**30 for r in runs],
        cap_reserved_gib=[r[1][name]['reserved']/2**30 for r in runs],
        labels_changed=sum(a!=b for a,b in zip(old['labels'],cap['labels'])))
Path('result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
