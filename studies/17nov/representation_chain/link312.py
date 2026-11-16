"""Link the dynamic §312 complete collector to immutable §315 consumers."""
import json
import os
import subprocess
from pathlib import Path

PRODUCER = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2')
CHAIN = Path('/mnt/lustre/work/gehler/gfh527/ni26_s315_312_20261117')
CONTROL_JOB = '3116073'


def main():
    jobs = json.loads((PRODUCER / 'wave2_jobs.json').read_text())
    collector = str(jobs['collector'])
    if not collector.isdecimal():
        raise ValueError('invalid complete-producer collector ID')
    current = subprocess.check_output(['squeue', '-u', 'gfh527', '-h', '-r', '-o', '%i'], text=True).splitlines()
    if len(current) + 2 > 272:
        raise RuntimeError('insufficient reserved submitted capacity; no consumer submitted')
    os.chdir(CHAIN)
    def submit(*args):
        return subprocess.check_output(['sbatch', '--parsable', *args], text=True).strip().split(';')[0]
    stage = submit(f'--dependency=afterok:{collector}', 'studies/17nov/representation_chain/stage.sbatch', '312')
    smoke = submit(f'--dependency=afterok:{stage}:{CONTROL_JOB}', 'studies/17nov/representation_chain/job.sbatch', 'smoke')
    with (CHAIN / 'consumer_jobs.json').open('x') as f:
        json.dump(dict(collector=collector, stage=stage, smoke=smoke), f)
    print(f'complete312collector{collector}→stage{stage}→zero/one-chain smoke{smoke}', flush=True)


if __name__ == '__main__':
    main()
