"""Use the unchanged branch guards with the six-child treatment batch."""
import importlib.util
import subprocess
import sys
from pathlib import Path

PATH = Path(__file__).with_name('post_smoke.py')
SPEC = importlib.util.spec_from_file_location('post_smoke', PATH)
POST = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(POST)
ORIGINAL_RUN = subprocess.run


def treatment_command(cmd):
    cmd = list(cmd)
    old = 'studies/16nov/lane1/chain_treatment.sbatch'
    if cmd and cmd[0] == 'sbatch' and old in cmd:
        if cmd.count(old) != 1:
            raise ValueError('unexpected treatment batch command')
        cmd[cmd.index(old)] = 'studies/16nov/lane1/chain_treatment_parallel.sbatch'
    return cmd


def run_with_parallel_batch(cmd, *args, **kwargs):
    return ORIGINAL_RUN(treatment_command(cmd), *args, **kwargs)


if __name__ == '__main__':
    POST.subprocess.run = run_with_parallel_batch
    POST.main(sys.argv[1])
