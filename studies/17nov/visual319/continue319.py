"""One-allocation-at-a-time, measured preflight-gated source-pilot continuation."""
import json
import os
import subprocess
import sys
from pathlib import Path


def main(stage):
    root = Path(os.environ['VISUAL_ROOT']).resolve(strict=True)
    report = json.loads((root / f'preflight_{stage}.json').read_text())
    if not report['passed'] or report['labels_opened']:
        raise ValueError('passed label-free preflight required')
    if stage == 'cuda':
        layout = report['layouts']['16']
        if max(v['median'] for v in layout['latency'].values()) > .15 or layout['gpu_peak_allocated'] > 8 * 1024**3:
            raise RuntimeError('measured bounded A100 cost/memory exceeds registered source-pilot gate')
        script = 'pilot.sbatch'
    elif stage == 'cpu':
        script = 'a100.sbatch'
    else:
        raise ValueError('unknown registered preflight phase')
    queued = subprocess.check_output(['squeue', '-u', 'gfh527', '-h', '-r', '-o', '%i'], text=True).splitlines()
    if len(queued) + 1 > 272:
        raise RuntimeError('submitted capacity unavailable;no continuation queued')
    parent = os.environ['SLURM_JOB_ID']
    job = subprocess.check_output(['sbatch', '--parsable', f'--dependency=afterok:{parent}', script], cwd=root, text=True).strip().split(';')[0]
    with (root / f'continuation_{stage}.json').open('x') as out:
        json.dump(dict(parent=parent, next_job=job, next_script=script, stage=stage), out, indent=2)
    print(f'319{stage}preflight passed;queued{script}job{job}afterallocation{parent}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
