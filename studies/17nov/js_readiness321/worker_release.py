"""Observe owning release, then condition worker probes on the complete six-seed collector."""
import json
import os
import subprocess
from pathlib import Path

from reference import sha
from proofs import manifest

ORCHESTRATION=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_orchestration4')


def submit(*args):return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0]


def main():
    ws=Path.cwd();manifest(ws)
    if (ws/'worker_jobs.json').exists():raise FileExistsError('worker jobs already released')
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%T'],text=True).splitlines()
    release=ORCHESTRATION/'released.json'
    if not release.exists() or len(lines)+2>244 or sum(s in ('RUNNING','COMPLETING') for s in lines)+2>32:
        count=int((ws/'deferrals.txt').read_text()) if (ws/'deferrals.txt').exists() else 0
        if count>=72:raise RuntimeError('bounded six-hour worker-release capacity wait expired')
        retry=submit('--dependency=singleton','--begin=now+5minutes','worker_release.sbatch')
        (ws/'deferrals.txt').write_text(str(count+1));print('capacity/owning-release deferred',retry,flush=True);return
    release=json.loads(release.read_text());collector=release['collector']
    ids={device:submit('--dependency=afterok:'+collector,f'worker_{device}.sbatch') for device in ('cpu','cuda')}
    (ws/'worker_jobs.json').write_text(json.dumps(dict(collector=collector,owning_release_sha256=sha(ORCHESTRATION/'released.json'),jobs=ids),indent=2))


if __name__=='__main__':main()
