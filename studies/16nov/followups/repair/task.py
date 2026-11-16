"""One §302 full-chain cell with six checked CP children."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGINAL = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes294_20261116/system')
ARMS = ('ET05', 'ET2', 'EPR25', 'EPR50', 'TI', 'PI', 'ZR')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')


def main(arm, order):
    if arm not in ARMS or order not in ORDERS or Path.cwd().resolve() != ROOT: raise ValueError('invalid cell')
    env = dict(os.environ, NM_NAME=arm, ADD='eegn', SWAP='none', ZERO='1' if arm == 'ZR' else '0', NM_SEEDS='0,1,2,3,4,5')
    subprocess.run([sys.executable, '-B', '-I', str(HERE / 'run.py'), order], env=env, check=True)
    path = ROOT / 'studies/15nov/s281/galv_v66_job.py'
    spec = importlib.util.spec_from_file_location('children', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    logdir = ROOT / 'logs/cp_302'; logdir.mkdir(parents=True, exist_ok=True)
    commands = [(f'{arm}_{order}_{b}', [sys.executable, '-B', '-I', str(HERE / 'cp.py'), arm, order, b]) for b in BASES]
    mod.run_children(commands, logdir, env)
    if arm in ('TI', 'PI'):
        for base in BASES:
            name = f'{arm}_{order}_{base}.npz'; reference = f'E05_{order}_{base}.npz'
            with np.load(ROOT / 'runs/s281_v66_cp' / name) as a, np.load(ORIGINAL / 'runs/s281_v66_cp' / reference) as b:
                if not np.array_equal(a['scores'], b['scores']): raise ValueError(f'{arm}/{base}: E05 CP identity failed')
        print(f'{arm}: all six CP outputs equal archived E05', flush=True)


if __name__ == '__main__': main(*sys.argv[1:])
