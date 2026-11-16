"""One ENU or fresh zero cell with six unchanged CP children."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPAIR = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes302_20261116/repair')
ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
BASES = ('B2_pool','B2_s345','B2_s012','B2_s0','B2_s1','B2_s2')


def main(arm, order):
    if arm not in ('ZR','ENU') or order not in ORDERS or Path.cwd().resolve() != ROOT: raise ValueError('invalid cell')
    env = dict(os.environ, NM_NAME=arm, ADD='eegn', SWAP='none', ZERO='1' if arm == 'ZR' else '0', NM_SEEDS='0,1,2,3,4,5')
    subprocess.run([sys.executable, '-B', '-I', str(HERE / 'run.py'), order], env=env, check=True)
    if arm == 'ZR':
        for pattern in ('runs/newmember/s281_v66p_{}_{}_gate.npz','runs/chainout/s281_v66p_{}_{}.npz'):
            with np.load(ROOT / pattern.format(arm,order)) as a, np.load(REPAIR / pattern.format(arm,order)) as b:
                if set(a.files) != set(b.files) or any(not np.array_equal(a[k],b[k]) for k in b.files):
                    raise ValueError('new-source zero chain differs from §302 zero')
    path = ROOT / 'studies/15nov/s281/galv_v66_job.py'
    spec = importlib.util.spec_from_file_location('children',path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    logdir = ROOT / 'logs/cp_neutral'; logdir.mkdir(parents=True, exist_ok=True)
    commands=[(f'{arm}_{order}_{b}',[sys.executable,'-B','-I',str(HERE/'cp.py'),arm,order,b]) for b in BASES]
    mod.run_children(commands,logdir,env)
    if arm == 'ZR':
        for base in BASES:
            name=f'ZR_{order}_{base}.npz'
            with np.load(ROOT/'runs/s281_v66_cp'/name) as a, np.load(REPAIR/'runs/s281_v66_cp'/name) as b:
                if not np.array_equal(a['scores'],b['scores']): raise ValueError('new-source zero CP differs')
        print('fresh zero chain and six CP outputs equal §302 zero',flush=True)


if __name__ == '__main__': main(*sys.argv[1:])
