"""Fail-closed bounded release after actual two-family source/timing smoke."""
import json
import os
import subprocess
from pathlib import Path

ws = Path(os.environ['R355_WS'])
root = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
for family in ('lag','envelope'):
    proof = json.loads((ws/'smoke'/family/'proof.json').read_text())
    assert proof['epochs'] == 2 and proof['labels_scored'] is False
    assert proof['projected_family_seconds'] <= 2880, proof
    assert proof['cuda_peak_bytes'] <= 8*1024**3 and proof['maxrss_kib'] <= 10*1024**2
    assert len(proof['folds']) == 1
raw = subprocess.check_output(['squeue','-u','gfh527','-h','-o','%i %T'],text=True).splitlines()
running = sum('RUNNING' in line or 'COMPLETING' in line for line in raw)
assert running + 2 <= 32 and len(raw) + 3 <= 244, (running,len(raw))
ids = []
for family in ('lag','envelope'):
    env = dict(os.environ, R355_FAMILY=family, R355_MODE='full')
    job = subprocess.check_output(['sbatch','--parsable','--time=00:50:00',
        '--output='+str(ws/f'{family}-%j.log'),str(ws/'code/job.sbatch')],env=env,text=True).strip()
    ids.append(job)
    (ws/'released.json').write_text(json.dumps(dict(producers=ids, capacity=raw),indent=2))
collector = subprocess.check_output(['sbatch','--parsable','--dependency=afterok:'+':'.join(ids),
    '--output='+str(ws/'collect-%j.log'),str(ws/'code/collect.sbatch')],text=True).strip()
(ws/'released.json').write_text(json.dumps(dict(producers=ids,collector=collector,capacity=raw),indent=2))
print((ws/'released.json').read_text(),flush=True)
