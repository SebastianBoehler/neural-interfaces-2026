import json
from pathlib import Path
import sys

phase = sys.argv[1]
result = {}
for mode in ('ordinary', 'stress'):
    result[mode] = []
    for pair in range(1 if phase == 'smoke' else 2):
        a, b = [json.loads(Path(f'{phase}-{mode}-{pair}-{arm}.json').read_text())
                for arm in ('stock', 'candidate')]
        for field in ('labels', 'activity', 'warm_activity'):
            if a[field] != b[field]:
                raise ValueError(f'{mode} {field}: stock differs from candidate')
        result[mode].append(dict(stock_s=a['s'], candidate_s=b['s'],
                                 saving_pct=100*(1-b['s']/a['s']),
                                 stock_wall_s=a['wall_s'], candidate_wall_s=b['wall_s'],
                                 activity=b['activity']))
Path(f'{phase}.json').write_text(json.dumps(result, indent=2) + '\n')
print(result, flush=True)
