"""Execute one registered §294 arm and all six existing CP variants."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from recipes import ARMS, get

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')


def main(arm, order):
    if (arm not in ARMS and arm != 'zero') or order not in ORDERS or Path.cwd().resolve() != ROOT:
        raise ValueError('invalid arm, order, or working directory')
    family, _, coral, _ = get(arm)
    add = ([] if family == 'none' else [family]) + (['fb_coral10'] if coral == 'add' else [])
    env = dict(os.environ, NM_NAME=arm, ADD=','.join(add) or 'none',
               SWAP='fb_coral10' if coral == 'swap' else 'none', ZERO='1' if arm == 'zero' else '0',
               NM_SEEDS='0,1,2,3,4,5')
    subprocess.run([sys.executable, '-B', '-I', str(HERE / 'run.py'), order], env=env, check=True)
    path = ROOT / 'studies/15nov/s281/galv_v66_job.py'
    spec = importlib.util.spec_from_file_location('children', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    logdir = ROOT / 'logs/cp_294'; logdir.mkdir(parents=True, exist_ok=True)
    commands = [(f'{arm}_{order}_{b}', [sys.executable, '-B', '-I', str(HERE / 'cp.py'), arm, order, b]) for b in BASES]
    module.run_children(commands, logdir, env)


if __name__ == '__main__':
    main(*sys.argv[1:])
