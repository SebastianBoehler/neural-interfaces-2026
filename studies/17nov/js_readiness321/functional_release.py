"""A new complete functional matrix only after an actual optimized first-stop timing failure."""
import json
import os
import subprocess
import sys
from pathlib import Path

from dispatch import submit
from proofs import manifest

PARENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_phase_dag1')


def main(variant,device):
    ws=Path.cwd();manifest(ws);manifest(PARENT)
    path=PARENT/f'output_{variant}_{device}/{device}_failure.json'
    decision=ws/f'decision_{variant}_{device}.json'
    if not path.exists():
        decision.write_text(json.dumps(dict(status='no_registered_timing_bar_failure_no_functional_release')));return
    r=json.loads(path.read_text())
    if r['status']!='worker_slack_guard_failed' or r['device']!=device or not r['no_later_rows_harvested'] or r['row']['ratio']<=r['allowed_ratio']:
        raise ValueError('actual registered timing failure not established')
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%T'],text=True).splitlines()
    if len(lines)+1>244 or sum(s in ('RUNNING','COMPLETING') for s in lines)+1>32:
        print('capacity deferred',submit('--dependency=singleton','--begin=now+5minutes','functional_release.sbatch',variant,device));return
    if decision.exists():raise FileExistsError('functional decision already recorded')
    job=submit(f'functional_{variant}_{device}.sbatch')
    decision.write_text(json.dumps(dict(status='separate_complete_functional_only_released',job=job,old_timing_failure_preserved=True),indent=2));print(job,flush=True)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
