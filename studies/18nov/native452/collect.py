"""Aggregate the complete pair of checkpoint banks, retaining every different decision."""
import json
from pathlib import Path

import score379 as ref


def read(seed, person):
    data = json.loads(Path(f'{seed}_{person}.json').read_text())
    assert data['seed'] == seed and data['person'] == person
    assert data['source'] == ref.sha(Path('source.sha256'))
    assert data['library'] == ref.sha(Path('decoder.so'))
    return data


rows = [read(s, p) for s in (379, 1441) for p in ref.PEOPLE]
assert sum(i['decisions'] for i in rows) == 2655*8*2*2
out = dict(pass_all=all(i['pass'] for i in rows), decisions=sum(i['decisions'] for i in rows),
           different=sum(len(i['mismatches']) for i in rows), max_abs=max(i['max_abs'] for i in rows),
           resets=sum(i['sessions'] for i in rows), reset_errors=sum(i['reset_errors'] for i in rows),
           history_label_changes=sum(i['changed_by_history'] for i in rows), folds=rows)
Path('result.json').write_text(json.dumps(out, indent=2)+'\n')
print({k:v for k,v in out.items() if k!='folds'})
