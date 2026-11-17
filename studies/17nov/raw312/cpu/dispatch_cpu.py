"""CPU-only waves; complete160cell collector, no partial accuracy or device mixing."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2')
TASKS = [i for i in range(160) if i != 80]
BATCHES = [TASKS[i:i + 8] for i in range(0, len(TASKS), 8)]


def main(ordinal):
    os.chdir(ROOT)
    probe = ROOT / 'output/fb_T_s0_P01.npz'
    if not probe.is_file():
        raise ValueError('complete guarded canonical CPU probe required')
    count = len(subprocess.check_output(['squeue', '-u', 'gfh527', '-h', '-r', '-o', '%i'], text=True).splitlines())
    def submit(*args):
        return subprocess.check_output(['sbatch', '--parsable', *args], text=True).strip().split(';')[0]
    if ordinal == len(BATCHES):
        if count + 1 > 272:
            raise RuntimeError('submitted capacity unavailable for collector')
        collector = submit('collect_cpu.sbatch')
        (ROOT / 'collector_job.json').write_text(json.dumps(dict(job=collector, device='cpu')))
        print(f'complete CPU matrix queued for owning collector{collector}', flush=True)
        return
    tasks = BATCHES[ordinal]
    running = len(subprocess.check_output(['squeue', '-u', 'gfh527', '-h', '-t', 'R', '-o', '%i'], text=True).splitlines())
    slots = min(4, 32 - running)
    if count + len(tasks) + 1 > 272 or slots < 1:
        raise RuntimeError('CPU wave capacity unavailable; nothing released')
    array = submit('--array=' + ','.join(map(str, tasks)) + f'%{slots}', 'work_cpu.sbatch', 'production')
    callback = submit(f'--dependency=afterok:{array}', 'dispatch_cpu.sbatch', str(ordinal + 1))
    with (ROOT / 'cpu_jobs.jsonl').open('a') as f:
        f.write(json.dumps(dict(wave=ordinal, array=array, callback=callback, tasks=tasks, slots=slots)) + '\n')
    print(f'CPU wave{ordinal}:array{array},callback{callback},concurrency{slots}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
