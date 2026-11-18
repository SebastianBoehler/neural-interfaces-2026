"""Produce the retained report from a validated collector's intermediate cells."""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

state = runpy.run_path('score.py')
delta, describe = state['delta'], state['describe']
result = state['res']
result['fixed_minus_full'] = describe(delta[..., 1] - delta[..., 0])
result['moved_minus_full'] = describe(delta[..., 2] - delta[..., 0])
rng = np.random.default_rng(1416)
days = rng.integers(2, size=(10000,2,1))
seeds = rng.integers(2, size=(10000,2,2))
primary = delta[...,1:].mean(-1)-delta[...,0]
result['person_blocked_minus_full'] = {
    person: dict(mean=values.mean(),
                 conditional_ci80=np.quantile(values[days,seeds].mean((1,2)), [.1,.9]).tolist())
    for person, values in zip(result['people'],primary)}
activity = state['activity']
result['activity_totals'] = {}
for kind in state['kinds']:
    bank = [d for k,d in activity.items() if k.startswith(kind+'|cap|')]
    result['activity_totals'][kind] = {key:sum(d[key] for d in bank)
                                       for key in ('windows','ft_refits','ft_steps','cp_refits','cp_fused','errors')}
    result['activity_totals'][kind]['recognized_sessions'] = sum(d['recognized'] >= 0 for d in bank)
    old = [d for k,d in activity.items() if k.startswith(kind+'|old|')]
    result['activity_totals'][kind]['old_routed'] = sum(d['routed'] for d in old)
result['input_digests'] = {str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest()
                           for path in ('sources.sha256','members.sha256','orders.npz')}
result['jobs'] = Path('jobs.txt').read_text().split()
Path('report.json').write_text(json.dumps(result,indent=2)+'\n')
