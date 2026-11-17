"""Release one component profile only for the actual A100 runtime-bar failure."""
import json
import os
import subprocess
from pathlib import Path

from proofs import manifest
from dispatch import submit

WORKER=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_worker2')


def main():
    ws=Path.cwd();manifest(ws);path=WORKER/'output_cuda/cuda_failure.json'
    if not path.exists():
        (ws/'decision.json').write_text(json.dumps(dict(status='no_gpu_runtime_bar_failure_profile_not_released')));return
    report=json.loads(path.read_text())
    if report['status']!='worker_slack_guard_failed' or report['device']!='cuda' or report['row']['ratio']<=report['allowed_ratio'] or not report['no_later_rows_harvested']:
        raise ValueError('actual failure is not the registered A100 timing stop')
    lines=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%T'],text=True).splitlines()
    if len(lines)+1>244 or sum(s in ('RUNNING','COMPLETING') for s in lines)+1>32:
        print('capacity deferred',submit('--dependency=singleton','--begin=now+5minutes','profile_release.sbatch'));return
    if (ws/'decision.json').exists():raise FileExistsError('profile decision already recorded')
    job=submit('profile.sbatch');(ws/'decision.json').write_text(json.dumps(dict(status='actual_gpu_timing_failure_component_profile_released',job=job),indent=2))


if __name__=='__main__':main()
