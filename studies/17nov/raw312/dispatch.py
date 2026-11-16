"""Capacity-bounded immutable DAG dispatch; no scoring until whole matrix completes."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ['NI26_WS'])


def submit(*args):
    return subprocess.check_output(['sbatch', '--parsable', *args], text=True).strip().split(';')[0]


def main():
    os.chdir(ROOT)
    # This controller runs after first80complete, releasing their queue slots.
    lines = subprocess.check_output(['squeue', '-u', 'gfh527', '-h', '-r', '-o', '%i'], text=True).splitlines()
    if len(lines) + 82 > 272:
        raise RuntimeError(f'need82available submitted slots; currently{len(lines)}; no production queued')
    if not (ROOT / 'guard_cuda.json').is_file() or not (ROOT / 'guard_cpu.json').is_file():
        raise ValueError('passed explicit CPU and GPU guards required')
    array = submit('--array=80-159%2', 'work.sbatch', 'production')
    collector = submit(f'--dependency=afterok:{array}', 'collect.sbatch')
    with (ROOT / 'wave2_jobs.json').open('x') as f:
        json.dump(dict(producer=array, collector=collector), f)
    print(f'queued wave2={array} whole collector={collector}', flush=True)


if __name__ == '__main__':
    main()
