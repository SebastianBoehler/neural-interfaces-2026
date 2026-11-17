"""Release the pre-registered bounded pilot only after actual v2 engineering success."""
import json
import subprocess
from pathlib import Path

from common import load, sha

GUARD=Path('/mnt/lustre/work/gehler/gfh527/ni26_spatial342_v2_20261117')
PILOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_spatial342_pilot_v2_20261117')
EXPECTED='93d996cdd21b69ba81632e534bd2264a7d5286d106ae5c45dac3a59d2fd9df90'


def main():
    if sha(PILOT/'source.sha256')!=EXPECTED:
        raise ValueError('prospective pilot source differs')
    guard=json.loads((GUARD/'gpu_guard.json').read_text())
    if not guard['passed'] or guard['P09_labels_opened'] or guard['P01_P02_P03_inference']:
        raise ValueError('actual source-only compatibility required')
    if not (GUARD/'random_initial.pt').exists() or not (GUARD/'same_weight_cpu_gpu_probe.npz').exists():
        raise ValueError('same-weight random/probe artifacts absent')
    cost=guard['records']['full_width_cost']
    if cost['d']!=65600 or cost['n']!=48 or cost['max_gradient']>1e-5:
        raise ValueError('full-width guard differs')
    _,_,source,query=load()
    # Pessimistic: every head as large as full65600; 5× observed optimizer work.
    projected=cost['seconds']*(len(source)/48)*5*32+2*(len(source)+len(query))*.015+120
    if projected>1200:
        raise ValueError('conservative measured projection exceeds 20-minute pilot')
    states=subprocess.check_output(['squeue','-u','gfh527','-h','-o','%T'],text=True).splitlines()
    if len(states)>=244 or sum(s.strip()=='RUNNING' for s in states)>=32:
        raise ValueError('account capacity unavailable; no automatic retry')
    job=subprocess.check_output(['sbatch','--parsable','pilot.sbatch'],cwd=PILOT,text=True).strip()
    record=dict(passed=True,pilot_job=job,projected_seconds=projected,actual_smoke=cost,
                guard_sha256=sha(GUARD/'gpu_guard.json'),pilot_source_sha256=EXPECTED,
                account_active_before=len(states),account_running_before=sum(s.strip()=='RUNNING' for s in states))
    Path('release.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
