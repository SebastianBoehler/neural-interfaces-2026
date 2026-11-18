"""Compare reversed streams against both immutable complete §387 runs."""
import json
from pathlib import Path

first = Path('../ni26_runtime387_v1_20261118')
result = {}
for arm in ('old','cap'):
    prior = [json.loads((first / f'{arm}_{rep}.json').read_text())['streams'] for rep in range(2)]
    rev = json.loads(Path(f'{arm}_0.json').read_text())['streams']
    assert list(rev) == ['P10-3','Schirrmeister2017/2','Schirrmeister2017/1']
    result[arm] = {}
    for key, record in rev.items():
        for regular in prior:
            original = regular[key]
            for field in ('rows','row_hash','labels','activity'):
                assert record[field] == original[field], (arm,key,field)
        times = [r[key]['seconds'] for r in prior]
        result[arm][key] = dict(original=times, reversed=record['seconds'],
                                delta=record['seconds']-sum(times)/len(times),
                                live_gib=record['allocated']/2**30, reserved_gib=record['reserved']/2**30)
Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
