"""One gated arm/order and six existing package-matched CP children."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARMS = {'EG06': 'eegn', 'EG08': 'eegn', 'DG06': 'eegn_dist2', 'DG08': 'eegn_dist2', 'ZG06': 'eegn'}
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')


def main(arm, order):
    if arm not in ARMS or order not in ORDERS or Path.cwd().resolve() != ROOT:
        raise ValueError('invalid gated arm, order or workspace')
    env = dict(os.environ, NM_NAME=arm, ADD=ARMS[arm], SWAP='none', ZERO='1' if arm == 'ZG06' else '0', NM_SEEDS='0,1,2,3,4,5')
    subprocess.run([sys.executable, '-B', '-I', str(HERE / 'run.py'), order], env=env, check=True)
    path = ROOT / 'studies/15nov/s281/galv_v66_job.py'
    spec = importlib.util.spec_from_file_location('children', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    logdir = ROOT / 'logs/cp_296'; logdir.mkdir(parents=True, exist_ok=True)
    children = [(f'{arm}_{order}_{b}', [sys.executable, '-B', '-I', str(HERE / 'cp.py'), arm, order, b]) for b in BASES]
    module.run_children(children, logdir, env)


if __name__ == '__main__':
    main(*sys.argv[1:])
