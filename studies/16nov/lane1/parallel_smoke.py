"""Exercise the existing six-child launcher on six real control artifacts."""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('s281_children', ROOT / 'studies/15nov/s281/galv_v66_job.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')

if __name__ == '__main__':
    if Path.cwd().resolve() != ROOT:
        raise ValueError('incorrect workspace')
    logdir = ROOT / 'logs/parallel_smoke'; logdir.mkdir(parents=True, exist_ok=False)
    code = ('import sys,numpy as np; z=np.load(sys.argv[1]); '
            'x=z["scores"]; assert x.shape==(2655,3) and np.isfinite(x).all(); print("REAL CONTROL OK")')
    cmds = [(b, [sys.executable, '-B', '-I', '-c', code,
                 str(ROOT / f'runs/s281_v66_cp/control_within0_{b}.npz')]) for b in BASES]
    M.run_children(cmds, logdir, dict(os.environ))
    print('six real-control children completed; no score computed', flush=True)
