"""Run six existing CP processes using the proven §281 child launcher."""
import importlib.util
import os
import sys
from pathlib import Path

BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')
ARMS = ('zero_eegn', 'E', 'D', 'C3A', 'C3S', 'C10A', 'C10S')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def commands(root, python, arm, order):
    if arm not in ARMS or order not in ORDERS:
        raise ValueError('invalid arm or order')
    cp = root / 'studies/14nov/newmember/s281_cp.py'
    return [(f'{arm}_{order}_{b}', [python, '-B', '-I', str(cp), arm, order, b]) for b in BASES]


def main(arm, order):
    root = Path(__file__).resolve().parents[3]
    if Path.cwd().resolve() != root:
        raise ValueError('run from the private workspace root')
    path = root / 'studies/15nov/s281/galv_v66_job.py'
    spec = importlib.util.spec_from_file_location('s281_children', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    logdir = root / 'logs/cp_parallel'; logdir.mkdir(parents=True, exist_ok=True)
    module.run_children(commands(root, sys.executable, arm, order), logdir, dict(os.environ))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
